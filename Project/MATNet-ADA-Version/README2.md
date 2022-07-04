## Motion-Attentive Transition for Zero-Shot Video Object Segmentation

This is a PyTorch implementation of MATNet for unsupervised video object segmentation.

__Motion-Attentive Transition for Zero-Shot Video Object Segmentation.__ [[Arxiv](https://arxiv.org/abs/2003.04253)] [[TIP](https://ieeexplore.ieee.org/document/9165947)]

## Prerequisites

The training and testing experiments are conducted using PyTorch 1.0.1 with a single GeForce RTX 2080Ti GPU with 11GB Memory.

- [PyTorch 1.0.1](https://github.com/pytorch/pytorch)

*Have patience :relaxed:, we'll go through one-by-one, first `training` pipeline and next `test` pipeline to imlpement this code fully. Note that we have shown results only on `DAVIS17` dataset.* 

We'll break this task into 5 modules:
1. Preparing Edge Annotations
2. Preparing HED Results
3. Generating Optical Flow
4. Training
5. Testing

*__Note__: From above 5 modules, first three are required for `training` while for `testing` 2nd and 3rd modules are only needed.*

#### Preparing Edge Annotations

Ensure that you have `matlab` installed in your system. Or if you are running it in some server try to do:
```bash
module load matlab/R20<version-available>
```
#### Preparing HED Results

For this make a separate pythonic environment (probably with `virtualenv` package) as suggested in `requirements_hed.txt` with the following configurations:

```python3
python3.8
numpy==1.22.3
Pillow==9.1.0
torch==1.11.0
typing_extensions==4.2.0
argparse==1.4.0
```

#### Preparing Optical Flow

Same goes here, but I would suggest to have a `conda` environment as we need `CuPY` here, that in turn installs `CUDA Toolkit` depending on the version of `CUDA` you have, and runs on `Python3.7.0` and above.

Follow the below steps to install it without any compatibility error and also load `CUDA` if you don't have activated it (check it using ```nvidia-smi```):

```bash
$ conda create -n myenv pyhton=3.8
$ module load cuda/10.2
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge cupy
$ pip install opencv-contrib-python
```
#### Requirements for MATNet

Also can be found in [requirements.txt](requirements.txt).

Can be installed by running

```bash
pip install -r requirements.txt
```

```python3
cycler==0.10.0
decorator==5.1.1
easydict==1.9
imageio==2.9.0
joblib==0.13.2
kiwisolver==1.1.0
lmdb==0.94
matplotlib==3.0.3
networkx==2.4
numpy==1.16.2
opencv-contrib-python==4.0.0.21
Pillow==7.2.0
pkg-resources==0.0.0
pydensecrf==1.0rc2
pyparsing==2.4.7
python-dateutil==2.8.2
PyWavelets==1.1.1
PyYAML==5.1.2
scikit-image==0.15.0
scipy==1.2.1
six==1.12.0
torch==1.0.1.post2
torchvision==0.2.2
tqdm==4.19.9
```

## Train

### Clone

```git clone --recursive https://github.com/rodosingh/MATNet.git; cd MATNet```

### Download Datasets

In the paper, we use the following publicly available dataset for training. Here are some steps to prepare the data:

[DAVIS-17](https://davischallenge.org/davis2017/code.html): However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training.

Do the following:
```mkdir data && cd data; ln -s <path-to-DAVIS17_data> DAVIS2017```

### Prepare Edge Annotations

Construct `edge-annotations` from given binary mask using matlab scripts. Chaange the path to `Annotation Folder` and also specify a `path-to-save-edge-annotations` inside script and run `run_davis2017.m` as follows (I would suggest to save it in the folder whose `soft-link` is `DAVIS2017`):
```bash
$ cd ../edge-annotations/
$ matlab
$ run_davis2017
```

### Prepare HED Results

To generate HED results for `DAVIS2017` follow the steps below:
```bash
$ source <path-to-hed-env>/bin/activate
$ cd <HED-Folder>
$ python run_davis.py --davis_folder <path-to-JPEGImages>/480p --save_dir <path-to-DAVIS2017-Folder>/DAVIS2017-HED
```
The codes are borrowed from <https://github.com/sniklaus/pytorch-hed>.

### Prepare Optical Flow

To generate optical flow results follow the steps mentioned below.
Prior to that `deactivate` if any environment is active.
```bash
$ conda activate <env-name>
$ cd <OPtical-Flow-Folder>
python run_davis_flow.py --davis_folder <path-to-JPEGImages>/480p --save_dir <path-to-DAVIS2017-Folder>/DavisFlow
```

The codes are borrowed from <https://github.com/sniklaus/pytorch-pwc>.

#-------------------------------------------------------------------------------

### Train on `DAVIS2017`

Ensure that all above steps are followed and required libraries are installed.<br>
Follow the steps below:

```bash
$ conda deactivate && deactivate
$ source <path-to-env-for-matnet>/bin/activate
$ cd python train_MATNet.py
```

## Test

Do the following in the same `pythonic env` which was used for training.

1. Run the following to obtain the saliency results on DAVIS2017 val set which can be found in `ImageSets/2017/val.txt` folder in `DAVIS2017`.

```bash
python test_MATNet.py --davis_result_dir <folder-to-save-results> --val_set <txt-file-containing-val-object-names> --davis_root_dir <path-to-folder-JPEGImages/480p> --davis_flow_dir <path-to-folder-of-flow-results>
``` 

2. Run for binary segmentation results.

```bash
python apply_densecrf_davis.py --image_dir <path-to-folder-JPEGImages/480p> --davis_result_dir <path-to-Davis-Results-folder>
```
### Test on your own Video

Run the following bash script within the same `env`.

```bash
$ ./test_vid.sh
```

## Segmentation Results

1. The segmentation results on DAVIS-16 and Youtube-objects can be downloaded from [Google Drive](https://drive.google.com/file/d/1d23TGBtrr11g8KFAStwewTyxLq2nX4PT/view?usp=sharing).
2. The segmentation results on DAVIS-17 __val__ can be downloaded from [Google Drive](https://drive.google.com/open?id=1GTqjWc7tktw92tBNKln2eFmb9WzdcVrz). We achieved __58.6__ in terms of _Mean J&F_.
3. The segmentation results on DAVIS-17 __test-dev__ can be downloaded from [Google Drive](https://drive.google.com/file/d/1Ood-rr0d4YRFSrGGh6yVpYvOvE_h0tVK/view?usp=sharing). We achieved __59.8__ in terms of _Mean J&F_. The method also achieved the second place in DAVIS-20 unsupervised object segmentation challenge. Please refer to [paper](https://davischallenge.org/challenge2020/papers/DAVIS-Unsupervised-Challenge-2nd-Team.pdf) for more details of our challenge solution.

## Pretrained Models

The pre-trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1XlenYXgQjoThgRUbffCUEADS6kE4lvV_/view?usp=sharing).
