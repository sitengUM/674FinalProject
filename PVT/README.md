# This code is largely based on PVT: Point-Voxel Transformer for 3D Deep Learning
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-voxel-transformer-an-efficient-approach/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=point-voxel-transformer-an-efficient-approach) 
which can be downloaded from [arXiv](https://arxiv.org/abs/2108.06076).

Since this code is a reimplementation, the code we authored is 

entirety of data/semantic3D/clean_data.py - written by Andrew

heavy modification of data/semantic3D/prepare_semantic.py to fit our need 
(base code is from original author's prepare_data.py but we tweaked almost everything)
-- Written by both Andrew and Siteng

model/semantic3Dpvt.py - tweaked in_channel & number of class - Andrew

modules/pvtconv.py - tweaked class SemPVTConv, added code in line 364-367 - Siteng

main_semantic3Dseg.py - tweaked default values & various other values. - Siteng and Andrew
Adjusted train/test code to work with semantic3D intended for s3dis. (Roughly 50-100 lanes change) 

data.py - line 234-294, added DataSet semantic3D to be used by DataLoader - Siteng 

utils.py - modified eps to have the loss not explode to nan. - Siteng





## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.3
- [numba](https://github.com/numba/numba)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [six](https://github.com/benjaminp/six)
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [plyfile](https://github.com/dranjan/python-plyfile)
- [h5py](https://github.com/h5py/h5py)
- [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)
- Ninja
- CUDA

A pretrained model is provided at checkpoints/semantic3Dseg/model_seg.t7.

## Example training and testing


```
#train
python main_semantic3Dseg.py 

#test
python main_semantic3Dseg.py  --eval=True

```

The original code and author is cited below, we reimplemented this code 
(with tweaks) to work with semantic3D
```
@article{zhang2021point,
  title={PVT: Point-Voxel Transformer for 3D Deep Learning},
  author={Zhang, Cheng and Wan, Haocheng and Liu, Shengqiang and Shen, Xinyi and Wu, Zizhao},
  journal={arXiv preprint arXiv:2108.06076},
  year={2021}
}
```
