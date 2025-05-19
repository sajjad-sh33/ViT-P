# OneFormer+ViT-P

This folder contains the code implementation for evaluating the ViT-P model, using the OneFormer model as the mask proposal generator.


## Installation

We use an evironment with the following specifications, packages and dependencies:

- Ubuntu 20.04.3 LTS
- Python 3.12.7
- conda 24.9.2
- [PyTorch v2.2.0](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.17.0](https://pytorch.org/get-started/previous-versions/)
- [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/releases/tag/v0.6)
- [NATTEN 0.15.1](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.15.1)

### Setup Instructions

- Create a conda environment
  
  ```bash
  conda create --name oneformer_ViT_P python=3.12 -y
  conda activate oneformer_ViT_P
  ```

- Install packages and other dependencies.

  ```bash
  git clone https://github.com/sajjad-sh33/ViT-P
  cd ViT-P/OneFormer

  # Install Pytorch
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

  # Install opencv (required for running the demo)
  pip3 install -U opencv-python

  # Install detectron2
  python tools/setup_detectron2.py

  # Install other dependencies
  pip3 install git+https://github.com/cocodataset/panopticapi.git
  pip3 install git+https://github.com/mcordts/cityscapesScripts.git
  pip3 install -r requirements.txt
  ```

- Setup wandb.

  ```bash
  # Setup wand
  pip3 install wandb
  wandb login
  ```

- Setup CUDA Kernel for MSDeformAttn. `CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

  ```bash
  # Setup MSDeformAttn
  cd oneformer/modeling/pixel_decoder/ops
  sh make.sh
  cd ../../../..
  ```

- Setup CUDA Kernel for DCNv3. Requires CUDA installed.

  ```bash
  # Setup DCNv3
  cd oneformer/modeling/backbone/ops_dcnv3
  sh make.sh
  cd ../../../..
  ```


  ## Evaluation

- You need to pass the value of `task` token. `task` belongs to [panoptic, semantic, instance].

- To evaluate a model's performance, use:

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 8 \
    --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>


    ## Results

![Results](images/plots.svg)

- &dagger; denotes the backbones were pretrained on ImageNet-22k.
- Pre-trained models can be downloaded following the instructions given [under tools](tools/README.md/#download-pretrained-weights).

### ADE20K

| Method | Backbone | Crop Size |  PQ   | AP   | mIoU <br> (s.s) | mIoU <br> (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    |  :---:    | :---: | :---:| :---:           | :---:               | :---:   |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 640&times;640 | 49.8 | 35.9 | 57.0 | 57.7 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | Swin-L<sup>&dagger;</sup> | 896&times;896 | 51.1 | 37.6 | 57.4 | 58.3 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | Swin-L<sup>&dagger;</sup> | 1280&times;1280 | 51.4 | 37.8 | 57.0 | 57.7 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k_1280x1280.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | ConvNeXt-L<sup>&dagger;</sup> | 640&times;640 | 50.0 | 36.2 | 56.6 | 57.4 | 220M | [config](configs/ade20k/convnext/oneformer_convnext_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 640&times;640 | 50.5 | 36.0 | 58.3 | 58.4 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 896&times;896 | 51.2 | 36.8 | 58.1 | 58.6 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 1280&times;1280 | 51.5 | 37.1 | 58.3 | 58.7 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer (COCO-Pretrained) | DiNAT-L<sup>&dagger;</sup> | 1280&times;1280 | 53.4 | 40.2 | 58.4 | 58.8 | 223M | [config](configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280_coco_pretrain.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) |
| OneFormer | ConvNeXt-XL<sup>&dagger;</sup> | 640&times;640 | 50.1 | 36.3 | 57.4 | 58.8 | 372M | [config](configs/ade20k/convnext/oneformer_convnext_xlarge_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_xl_oneformer_ade20k_160k.pth) |
| OneFormer (emb_dim=256) | InternImage-H<sup>&dagger;</sup> | 896&times;896 | 54.5 | 40.2 | 60.4 | 60.8 | 1.10B | [config](configs/ade20k/intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) |
| OneFormer (emb_dim=1024, COCO-Pretrained) | InternImage-H<sup>&dagger;</sup> | 896&times;896 | 55.5 | 44.2 | 60.7 | 60.7 | 1.35B | [config](configs/ade20k/coco_pretrain_intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) |

### Cityscapes

| Method | Backbone |  PQ   | AP   | mIoU <br> (s.s) | mIoU <br> (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---: | :---:| :---:      | :---:          | :---:   |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 67.2 | 45.6 | 83.0 | 84.4 | 219M | [config](configs/cityscapes/swin/oneformer_swin_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth) |
| OneFormer | ConvNeXt-L<sup>&dagger;</sup> | 68.5 | 46.5 | 83.0 | 84.0 | 220M | [config](configs/cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_l_oneformer_cityscapes_90k.pth) |
| OneFormer (Mapillary Vistas-Pretrained) | ConvNeXt-L<sup>&dagger;</sup> | 70.1 | 48.7 | 84.6 | 85.2 | 220M | [config](configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_l_oneformer_cityscapes_90k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/mapillary/mapillary_pretrain_250_16_convnext_l_oneformer_mapillary_300k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 67.6 | 45.6 | 83.1 | 84.0 | 223M | [config](configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth) |
| OneFormer | ConvNeXt-XL<sup>&dagger;</sup> | 68.4 | 46.7 | 83.6 | 84.6 | 372M | [config](configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_xl_oneformer_cityscapes_90k.pth) |
| OneFormer (Mapillary Vistas-Pretrained) | ConvNeXt-XL<sup>&dagger;</sup> | 69.7 | 48.9 | 84.5 | 85.8 | 372M | [config](configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_xlarge_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/mapillary/mapillary_pretrain_250_16_convnext_xl_oneformer_mapillary_300k.pth) |
| OneFormer (emb_dim=256) | InternImage-H<sup>&dagger;</sup> | 70.6 | 50.6 | 85.1 | 85.7 | 1.10B | [config](configs/cityscapes/intern_image/oneformer_intern_image_huge_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_intern_image_h_oneformer_cityscapes_90k.pth) |

### COCO

| Method | Backbone |  PQ   |  PQ<sup>Th</sup>   |  PQ<sup>St</sup>   | AP | mIoU | #params | config | Checkpoint |
|   :---:| :---:    | :---: | :---:              | :---:              |:---:| :---:| :---:  |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 57.9 | 64.4 | 48.0 | 49.0 | 67.4 | 219M | [config](configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 58.0 | 64.3 | 48.4 | 49.2 | 68.1 | 223M | [config](configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) |
| OneFormer (emb_dim=1024) | InternImage-H<sup>&dagger;</sup> | 60.0 | 67.1 | 49.2 | 52.0 | 68.8 | 1.35B | [config](configs/coco/intern_image/oneformer_intern_image_huge_bs16_100ep_1024.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/250_16_intern_image_h_oneformer_coco_100ep_1024.pth) |
