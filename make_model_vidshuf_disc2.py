from email.policy import strict
import torch
import torch.nn as nn
import copy
#from vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID,Block
from vit_ID  import TransReID,Block
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    print('feature_random0',feature_random.shape)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def shuffle_unit_vid(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    
    
    
    #video shift
    #features=features.view(batchsize,4,-1)
    
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    
    
    '''
    begin=768
    feature_random = torch.cat([features[:, 0:768],features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    print('feature_random2',feature_random.shape)
    begin=(768*2)-shift
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    print('feature_random3',feature_random.shape)
    '''
    x = feature_random
    

   
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)
    
    x = torch.transpose(x, 1, 2).contiguous()
    
    x = x.view(batchsize, -1, dim)
    
    return x    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone11111111111111111111111'.format(cfg.MODEL.TRANSFORMER_TYPE))

       
        camera_num = camera_num
        
        
        self.base =TransReID(
        img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera_num,  drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,norm_layer=partial(nn.LayerNorm, eps=1e-6),  cam_lambda=3.0)
        #self.base = vit_base_patch16_224_TransReID(img_size=[256, 128], sie_xishu=3.0, local_feature=True, camera=camera_num, view=0, stride_size=[16, 16], drop_path_rate=0.1)#,overlap=False)
        state_dict = torch.load('/home2/zwjx97/TransReID-main/figs/transformerlabelsmothandcenter_Mars_model.pth', map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
                if('base' in k):
                       name=k[5:]
                       #print(name)
                       new_state_dict[name] = v
            #print(new_state_dict)
            
        self.base.load_param(new_state_dict,load=True)  
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384

       
            

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            print(i)
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num):
        super(build_transformer_local, self).__init__()
        
        self.in_planes = 768

      
        
        camera_num = camera_num
        
        #self.base =self.base = vit_base_patch16_224_TransReID(img_size=[256, 128], sie_xishu=3.0, local_feature=True, camera=camera_num, view=0, stride_size=[16, 16], drop_path_rate=0.1)#,overlap=False)
        from functools import partial
        self.base =TransReID(
        img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera_num,  drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,norm_layer=partial(nn.LayerNorm, eps=1e-6),  cam_lambda=3.0)
        #state_dict = torch.load('/home2/zwjx97/TransReID-main/figs/transformerlabelsmothandcenter_Mars_model.pth', map_location='cpu')

        state_dict = torch.load('/home2/zwjx97/TransReID-main/jx_vit_base_p16_224-80ecf9dd.pth', map_location='cpu')
        self.base.load_param(state_dict,load=True)

        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
                if('base' in k):
                       name=k[5:]
                       #print(name)
                       new_state_dict[name] = v
            #print(new_state_dict)
            
        self.base.load_param(new_state_dict,load=True) 
        '''   
            
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]  # stochastic depth decay rule
        
        self.block1 = Block(
                dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0, drop_path=dpr[11], norm_layer=partial(nn.LayerNorm, eps=1e-6))
       
       
       
        block= self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            self.block1,
            nn.LayerNorm(3072)#copy.deepcopy(layer_norm)
        )
        
        self.num_classes = num_classes
        
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        #-------------------video part-------------
        self.middle_dim = 256 # middle layer dimension
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1,1]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 

        #--------------------video shufl--------
        self.attention_conv_shufl = nn.Conv2d(3072, self.middle_dim, [1,1]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv_shufl = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 

        #------------------------------------------
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(3072)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(3072)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(3072)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(3072)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = 2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = 5
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = 4
        print('using divide_length size:{}'.format(self.divide_length))
        
        self.rearrange = True



    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        b=x.size(0)
        t=x.size(1)
        x=x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        
        features = self.base(x, cam_label=cam_label)#, view_label=view_label)
        #print('new shape features',features.shape)
        
        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        #features=features.view(b,features.size(1),t*features.size(2))
        #print('new shape features',features.shape)
        #print('self.block1',self.block1)
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        

        if self.rearrange:
            #x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
            
            features=features.view(b,features.size(1),t*features.size(2))
            x =shuffle_unit_vid(features, self.shift_num, self.shuffle_groups)
            #print('x end shape',x.shape)
        else:
            x = features[:, 1:]
        token = features[:, 0:1]    
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]
        #print('local_feat_4',local_feat_4.shape)
        from torch.nn import functional as F
        #-----------global_feat----------------
        t=4
        b=global_feat.size(0)//4
        #print('global_feat',global_feat.shape)
        global_feat=global_feat.unsqueeze(dim=2)
        global_feat=global_feat.unsqueeze(dim=3)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a 
        #x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = global_feat.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        global_feat = att_x.view(b,self.in_planes)
        #print('f',global_feat.shape)
        feat = self.bottleneck(global_feat)
        
        '''
        #-------------local_feat_1-----------------------
        t=4
        b=local_feat_1.size(0)//4
        #print('local_feat_1',local_feat_1.shape)
        local_feat_1=local_feat_1.unsqueeze(dim=2)
        local_feat_1=local_feat_1.unsqueeze(dim=3)
        a = F.relu(self.attention_conv_shufl(local_feat_1))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv_shufl(a))
        a = a.view(b, t)
        #x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = local_feat_1.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, 3072)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        local_feat_1= att_x.view(b,3072)
        
        local_feat_1_bn = self.bottleneck_1(local_feat_1)

        #-------------local_feat_2-----------------------
        t=4
        b=local_feat_2.size(0)//4
        #print('local_feat_2',local_feat_2.shape)
        local_feat_2=local_feat_2.unsqueeze(dim=2)
        local_feat_2=local_feat_2.unsqueeze(dim=3)
        a = F.relu(self.attention_conv_shufl(local_feat_2))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv_shufl(a))
        a = a.view(b, t)
        #x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = local_feat_2.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, 3072)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        local_feat_2= att_x.view(b,3072)

        local_feat_2_bn = self.bottleneck_2(local_feat_2)

        #-------------local_feat_3-----------------------
        t=4
        b=local_feat_3.size(0)//4
        #print('local_feat_3',local_feat_3.shape)
        local_feat_3=local_feat_3.unsqueeze(dim=2)
        local_feat_3=local_feat_3.unsqueeze(dim=3)
        a = F.relu(self.attention_conv_shufl(local_feat_3))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv_shufl(a))
        a = a.view(b, t)
        #x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = local_feat_3.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, 3072)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        local_feat_3= att_x.view(b,3072)


        local_feat_3_bn = self.bottleneck_3(local_feat_3)

        #-------------local_feat_4-----------------------
        t=4
        b=local_feat_4.size(0)//4
        #print('local_feat_4',local_feat_4.shape)
        local_feat_4=local_feat_4.unsqueeze(dim=2)
        local_feat_4=local_feat_4.unsqueeze(dim=3)
        a = F.relu(self.attention_conv_shufl(local_feat_4))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv_shufl(a))
        a = a.view(b, t)
        #x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = local_feat_4.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t,3072)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        local_feat_4= att_x.view(b,3072)

        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        '''
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        #print('local_feat_4 last ',local_feat_4.shape)
        if self.training:
            
            cls_score = self.classifier(feat)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
                
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4 ], [global_feat, local_feat_1, local_feat_2, local_feat_3,local_feat_4],  a_vals  # global feature for triplet loss
            #return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4 ], [feat, local_feat_1_bn , local_feat_2_bn , local_feat_3_bn, local_feat_4_bn ],  a_vals  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn/4 , local_feat_2_bn/4 , local_feat_3_bn /4, local_feat_4_bn/4 ], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path,load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
               self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
               print('Loading pretrained model from {}'.format(trained_path))
        else:
            param_dict=trained_path
            for i in param_dict:
             #print(i)   
             if i not in self.state_dict() or 'classifier' in i or 'sie_embed' in i:
                continue
             self.state_dict()[i].copy_(param_dict[i])
            '''
            for i in trained_path:
               print('i:',i)
               self.state_dict()[i].copy_(trained_path[i])
            '''  
            
           
        

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

'''
__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    
}
'''

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
            '''
            model_path='transformerlabelsmothandcenter_Mars_model.pth'
            state_dict = torch.load('transformerlabelsmothandcenter_Mars_model.pth', map_location='cpu')
            print('****************load mars ************************************')
            model.load_param(state_dict,load=True)
            
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
            '''
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model