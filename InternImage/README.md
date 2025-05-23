# InternImage+ViT-P

This folder contains the code implementation for evaluating the ViT-P model, using the InternImage model as the mask proposal generator.


## Installation

- Clone this repository:

```bash
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.9
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install `torch==1.11` with `CUDA==11.3`:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip.

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm`, `mmcv-full` and \`mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
# Please use a version of numpy lower than 2.0
pip install numpy==1.26.4
pip install pydantic==1.10.13
```

- Compile CUDA operators

Before compiling, please use the `nvcc -V` command to check whether your `nvcc` version matches the CUDA version of PyTorch.

```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

- You can also install the operator using precompiled `.whl` files
  [DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

## Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Evaluation

To evaluate a model's performance, use:

```bash
python test.py --config configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
       --Segmentation_WEIGHTS pretrained/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.pth \
       --Classification_WEIGHTS model_COCOStuff_711_200point.pth --eval mIoU
```


### ADE20K

| Method | Segmentaion Backbone | Classification Backbone | Segmentation resolution  | mIoU (s.s) | mIoU (ms+flip) | #params | config | Checkpoint |
|  :---:    |  :---:    | :---: | :---:| :---:           | :---:               | :---:   |  :---: |    :---:   |
| Mask2Former(COCO-Pretrained)+ViT-P | InternImage-H |DinoV2-L | 896&times;896 | 63.1 | 63.6 | 1.61B | [config](configs/ade20k/intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_ADE20k82_200point_cocopretrain.pth) |


### Cityscapes

| Method | Segmentaion Backbone	 | Classification Backbone| Segmentation resolution | mIoU (s.s) | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---: | :---:| :---:| :---:      | :---:          | :---:   |  :---: |    :---:   |
| Mask2Former(Mapillary-Pretrained)+ViT-P | InternImage-H |DinoV2-L | 1024&times;1024 | 86.8 | 87.4 | 1.4B | [config](configs/ade20k/intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_Cityscapes871_100point_cocopretrain.pth) |

### COCO-Stuff-164K

| Method | Segmentaion Backbone	 | Classification Backbone| Segmentation resolution | mIoU (s.s) | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---: | :---:| :---:| :---:      | :---:          | :---:   |  :---: |    :---:   |
| Mask2Former+ViT-P | InternImage-H |DinoV2-L | 896&times;896 | 53.5 | 53.7 | 1.61B | [config](configs/ade20k/intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_COCOStuff_711_200point.pth) |

