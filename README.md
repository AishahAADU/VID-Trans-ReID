# VID-Trans-ReID
This is an Official Pytorch Implementation of our paper VID-Trans-ReID: Enhanced Video Transformers for Person Re-identification

[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) Tested using Python 3.7.x and Torch: 1.8.0.

## Architecture:
<p align="center">
<img  width="811" alt="modelupdated2" src="https://user-images.githubusercontent.com/92983150/200542326-0d6fd560-f598-43ed-a812-d15eb1df77cf.png">
</p>

## Abstract
_"Video-based person Re-identification (Re-ID) has received increasing attention recently due to its important role within surveillance video analysis. Video-based Re-ID expands upon earlier image-based methods by extracting person features temporally across multiple video image frames. The key challenge within person Re-ID is extracting a robust feature representation that is invariant to the challenges of pose and illumination variation across multiple camera viewpoints. Whilst most contemporary methods use a CNN based methodology, recent advances in vision transformer (ViT) architectures boos fine-grained feature discrimination via the use of both multi-head attention without any loss of feature robustness. To specifically enable ViT architectures to effectively address the challenges of video person Re-ID, we propose two novel modules constructs, Tem- poral Clip Shift and Shuffled (TCSS) and Video Patch Part Feature (VPPF), that boost the robustness of the resultant Re-ID feature representation. Furthermore, we combine our proposed approach with current best practices spanning both image and video based Re-ID including camera view embedding. Our proposed approach outperforms existing state-of-the-art work on the MARS, PRID2011, and iLIDS-VID Re-ID benchmark datasets achieving 96.36%, 96.63%, 94.67% rank-1 accuracy respectively and achieving 90.25% mAP on MARS."_

[[A. Alsehaim, T.P. Breckon, In Proc. British Machine Vision Conference, BMVA, 2022] (https://breckon.org/toby/publications/papers/alsehaim22vidtransreid.pdf)] [[Talk](https://www.youtube.com/embed/NARrZroYD-U)] [[Poster](https://breckon.org/toby/publications/posters/alsehaim22vidtransreid_poster.pdf)]

##

<img  alt="non-id2" src="https://user-images.githubusercontent.com/92983150/200541008-577555d0-b61d-4609-b8d5-5aa6be3f08a8.png">

##

<img width="811" alt="paper2Dig" src="https://user-images.githubusercontent.com/92983150/198893163-0673a748-e2f1-4cd2-a2d6-f491ac5ddeae.gif">




## Requirements
```
pip install -r requirements.txt
```
## Getting Started

1. Download the ImageNet pretrained transformer model : [ViT_base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).
2. Download the video person Re-ID datasets [MARS](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html), [PRID](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid11/) and [iLIDS-VID](https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html)

## Train and Evaluate 

Use the pre-trained model [ViT_base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) to initialize ViT transformer then train the whole model. 

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

## Test
To test the model you can use our pretrained model on MARS dataset [download]()

```
python -u VID_Test.py --Dataset_name 'Mars' --model_path 'MarsMain_Model.pth'
```

## Acknowledgement
Thanks to Hao Luo, using some implementation from his [repository](https://github.com/michuanhaohao)

## Citation

If you are making use of this work in any way, you must please reference the following paper in any report, publication, presentation, software release or any other associated materials:

[VID-Trans-ReID: Enhanced Video Transformers for Person Re-identification](https://breckon.org/toby/publications/papers/alsehaim22vidtransreid.pdf) (A. Alsehaim, T.P. Breckon), In Proc. British Machine Vision Conference, BMVA, 2022. 

```
@inproceedings{alsehaim22vidtransreid,
 author = {Alsehaim, A. and Breckon, T.P.},
 title = {VID-Trans-ReID: Enhanced Video Transformers for Person Re-identification},
 booktitle = {Proc. British Machine Vision Conference},
 year = {2022},
 month = {November},
 publisher = {BMVA},
 url = {https://breckon.org/toby/publications/papers/alsehaim22vidtransreid.pdf}
}
```
