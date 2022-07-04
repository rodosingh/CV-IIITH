## Motion-Attentive Transition for Zero-Shot Video Object Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motion-attentive-transition-for-zero-shot/unsupervised-video-object-segmentation-on)](https://paperswithcode.com/sota/unsupervised-video-object-segmentation-on?p=motion-attentive-transition-for-zero-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/matnet-motion-attentive-transition-network/unsupervised-video-object-segmentation-on-4)](https://paperswithcode.com/sota/unsupervised-video-object-segmentation-on-4?p=matnet-motion-attentive-transition-network)

> UPDATES:<br>
> - [2021/04/17] Our MATNet achieves state-of-the-art results (__64.2__ in terms of _Mean J_) on the [MoCA](https://drive.google.com/drive/folders/1x-owzr9Voz65NQghrN_H1LEYDaaQP5n1?usp=sharing) dataset in "Self-supervised Video Object Segmentation by Motion Grouping" by Charig Yang, Hala Lamdouar, Erika Lu, Andrew Zisserman, Weidi Xie. Thanks [Charig Yang](https://charigyang.github.io/) for providing the segmentation results [Google Drive](https://drive.google.com/drive/folders/1x-owzr9Voz65NQghrN_H1LEYDaaQP5n1?usp=sharing).
> - [2020/06/15] Update results for DAVIS-17 test-dev set!
> - [2020/03/04] Update results for DAVIS-17 validation set!
> - [2019/11/17] Codes released!

This is a PyTorch implementation of our MATNet for unsupervised video object segmentation.

**Motion-Attentive Transition for Zero-Shot Video Object Segmentation.** [[Arxiv](https://arxiv.org/abs/2003.04253)] [[TIP](https://ieeexplore.ieee.org/document/9165947)]

## Prerequisites

The training and testing experiments are conducted using PyTorch 1.0.1 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
- [PyTorch 1.0.1](https://github.com/pytorch/pytorch)
                   
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

## Train

### Clone
```git clone --recursive https://github.com/tfzhou/MATNet.git```

### Download Datasets
In the paper, we use the following two public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. 
    However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training. 
- [YoutubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 10 frames in YoutubeVOS-2018. We use the dataset version with 6fps rather than 30fps.
- Create soft links:

    ```cd data; ln -s your/davis17/path DAVIS2017; ln -s your/youtubevos/path YouTubeVOS_2018;```
    
### Prepare Edge Annotations
I have provided some matlab scripts to generate edge annotations from mask. Please run ```data/run_davis2017.m``` 
and ```data/run_youtube.m```.

### Prepare HED Results
I have provided the pytorch codes to generate HED results for the two datasets (see ```3rdparty/pytorch-hed```).
Please run ```run_davis.py``` and ```run_youtube.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-hed. 

### Prepare Optical Flow
I have provided the pytorch codes to generate optical flow results for the two datasets (see ```3rdparty/pytorch-pwc```).
Please run ```run_davis_flow.py``` and ```run_youtubevos_flow.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-pwc. 
Please follow the [setup](https://github.com/sniklaus/pytorch-pwc#setup) section to install ```cupy```. 

`warning: Total size of optical flow results of Youtube-VOS is more than 30GB.`

### Train
Once all data is prepared, please run ```python train_MATNet.py``` for training.

## Test
1. Run ```python test_MATNet.py``` to obtain the saliency results on DAVIS-16 val set.
2. Run ```python apply_densecrf_davis.py``` for binary segmentation results.


## Segmentation Results

1. The segmentation results on DAVIS-16 and Youtube-objects can be downloaded from [Google Drive](https://drive.google.com/file/d/1d23TGBtrr11g8KFAStwewTyxLq2nX4PT/view?usp=sharing).
2. The segmentation results on DAVIS-17 __val__ can be downloaded from [Google Drive](https://drive.google.com/open?id=1GTqjWc7tktw92tBNKln2eFmb9WzdcVrz). We achieved __58.6__ in terms of _Mean J&F_.
3. The segmentation results on DAVIS-17 __test-dev__ can be downloaded from [Google Drive](https://drive.google.com/file/d/1Ood-rr0d4YRFSrGGh6yVpYvOvE_h0tVK/view?usp=sharing). We achieved __59.8__ in terms of _Mean J&F_. The method also achieved the second place in DAVIS-20 unsupervised object segmentation challenge. Please refer to [paper](https://davischallenge.org/challenge2020/papers/DAVIS-Unsupervised-Challenge-2nd-Team.pdf) for more details of our challenge solution.

## Pretrained Models

The pre-trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1XlenYXgQjoThgRUbffCUEADS6kE4lvV_/view?usp=sharing).

## Citation
If you find MATNet useful for your research, please consider citing the following papers:
```
@inproceedings{zhou2020motion,
  title={Motion-Attentive Transition for Zero-Shot Video Object Segmentation},
  author={Zhou, Tianfei and Wang, Shunzhou and Zhou, Yi and Yao, Yazhou and Li, Jianwu and Shao, Ling},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020},
  pages={13066--13073}
}

@article{zhou2020matnet,
  title={MATNet: Motion-Attentive Transition Network for Zero-Shot Video Object Segmentation},
  author={Zhou, Tianfei and Li, Jianwu and Wang, Shunzhou and Tao, Ran and Shen, Jianbing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={8326-8338},
  year={2020}
}

@inproceedings{zhou2021unsupervised,
  author = {Zhou, Tianfei and Li, Jianwu and Li, Xueyi and Shao, Ling},
  title = {Target-Aware Object Discovery and Association for Unsupervised Video Multi-Object Segmentation},
  booktitle = {CVPR},
  year = {2021}
}
```


