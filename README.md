# VID-Trans-ReID
This is an Official Pytorch Implementation of our paper VID-Trans-ReID: Enhanced Video Transformers for Person Re-identification

[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) Tested using Python 3.7.x and Torch: 1.8.0.

## Abstract
Video-based person Re-identification (Re-ID) has received increasing attention recently due to its important role within surveillance video analysis. Video-based Re-ID expands upon earlier image-based methods by extracting person features temporally across multiple video image frames. The key challenge within person Re-ID is extracting a robust feature representation that is invariant to the challenges of pose and illumination variation across multiple camera viewpoints. Whilst most contemporary methods use a CNN based methodology, recent advances in vision transformer (ViT) architectures boos fine-grained feature discrimination via the use of both multi-head attention without any loss of feature robustness. To specifically enable ViT architectures to effectively address the challenges of video person Re-ID, we propose two novel modules constructs, Tem- poral Clip Shift and Shuffled (TCSS) and Video Patch Part Feature (VPPF), that boost the robustness of the resultant Re-ID feature representation. Furthermore, we combine our proposed approach with current best practices spanning both image and video based Re-ID including camera view embedding. Our proposed approach outperforms existing state-of-the-art work on the MARS, PRID2011, and iLIDS-VID Re-ID benchmark datasets achieving 96.36%, 96.63%, 94.67% rank-1 accuracy respectively and achieving 90.25% mAP on MARS.

## Architectur:
<img width="811" alt="paper2Dig" src="https://user-images.githubusercontent.com/92983150/198893163-0673a748-e2f1-4cd2-a2d6-f491ac5ddeae.gif">

## Getting Started
1. Download the ImageNet pretrained transformer model : [ViT_base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).
2. Download the video person Re-ID datasets [MARS](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html), [PRID](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid11/) and [iLIDS-VID](https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html)
## Train and Evaluate 
MARS Datasete
```
python -u VID_Trans_ReID.py --Dataset_name 'Mars' --ViT_path 'jx_vit_base_p16_224-80ecf9dd.pth'
```

PRID Dataset
```
python -u VID_Trans_ReID.py --Dataset_name 'PRID' --ViT_path 'jx_vit_base_p16_224-80ecf9dd.pth'
```

iLIDS-VID Dataset
```
python -u VID_Trans_ReID.py --Dataset_name 'iLIDSVID' --ViT_path 'jx_vit_base_p16_224-80ecf9dd.pth'
```



## Cite
```
@inproceedings{alsehaim22vidtransreid,
 author = {Alsehaim, A. and Breckon, T.P.},
 title = {VID-Trans-ReID: Enhanced Video Transformers for Person Re-identification},
 booktitle = {Proc. British Machine Vision Conference},
 year = {2022},
 month = {November},
 publisher = {BMVA},
 keywords = {transformers, Re-ID, multi-camera, person reidentification, camera-to-camera tracking, deep learning},
 url = {https://breckon.org/toby/publications/papers/alsehaim22vidtransreid.pdf},
 category = {surveillance},
}
```
