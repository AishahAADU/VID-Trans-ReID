from Dataloader import dataloader
from VID_Trans_model import VID_Trans


from Loss_fun import make_loss

import random
import torch
import numpy as np
import os
import argparse

import logging
import os
import time
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torch.cuda import amp
import torch.distributed as dist

from utility import AverageMeter, optimizer,scheduler



   
        

       
from torch.autograd import Variable              
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
      for batch_idx, (imgs, pids, camids,_) in enumerate(queryloader):
       
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        
        b,  s, c, h, w = imgs.size()
        
        
        features = model(imgs,pids,cam_label=camids )
       
        features = features.view(b, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        
        q_pids.append(pids)
        q_camids.extend(camids)
      qf = torch.stack(qf)
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)
      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      for batch_idx, (imgs, pids, camids,_) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, s,c, h, w = imgs.size()
        features = model(imgs,pids,cam_label=camids)
        features = features.view(b, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.append(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve r1:",cmc[0])
    
    return cmc[0], mAP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name)
    model = VID_Trans( num_classes=num_classes, camera_num=camera_num,pretrainpath=pretrainpath)
    
    loss_fun,center_criterion= make_loss( num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= 0.5)
    
    optimizer= optimizer( model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()

    #Train
    device = "cuda"
    epochs = 120
    model=model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    cmc_rank1=0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        
        scheduler.step(epoch)
        model.train()
        
        for Epoch_n, (img, pid, target_cam,labels2) in enumerate(train_loader):
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            
            labels2=labels2.to(device)
            with amp.autocast(enabled=True):
                target_cam=target_cam.view(-1)
                score, feat ,a_vals= model(img, pid, cam_label=target_cam)
                
                labels2=labels2.to(device)
                attn_noise  = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()
                
                loss_id ,center= loss_fun(score, feat, pid, target_cam)
                loss=loss_id+ 0.0005*center +attn_loss
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            for param in center_criterion.parameters():
                    param.grad.data *= (1. / 0.0005)
            scaler.step(optimizer_center)
            scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (Epoch_n + 1) % 50 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (Epoch_n + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        if (epoch+1)%10 == 0 :
               
               model.eval()
               cmc,map = test(model, q_val_set,g_val_set)
               print('CMC: %.4f, mAP : %.4f'%(cmc,map))
               if cmc_rank1 < cmc:
                  cmc_rank1=cmc
                  torch.save(model.state_dict(),os.path.join('/home2/zwjx97/VID-Trans-ReID',  Dataset_name+'Main_Model.pth')) 
        
     
