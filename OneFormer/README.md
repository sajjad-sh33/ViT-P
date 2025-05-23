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

- Setup CUDA Kernel for DCNv3. Requires CUDA installed.

  ```bash
  # Setup DCNv3
  cd oneformer/modeling/backbone/ops_dcnv3
  sh make.sh
  cd ../../../..
  ```

- To execute the commands provided in the next section for evaluation, the dinov2 package must be included in the Python module search path.

  ```bash
  cd ../ViT-P
  export PYTHONPATH="$PYTHONPATH:$(pwd)"
  cd ../OneFormer
  ```


## Dataset Preparation

- We experiment on three major benchmark dataset: ADE20K, Cityscapes and COCO 2017.
- Please see [Preparing Datasets for OneFormer](datasets/README.md) for complete instructions for preparing the datasets.

 ## Evaluation

- You need to pass the value of `task` token. `task` belongs to [panoptic, semantic, instance].

- To evaluate a model's performance, use:

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 1 \
    --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-segmentation-checkpoint> \
    MODEL.Classification_WEIGHTS <path-to-Classification-checkpoint> \
    MODEL.TEST.TASK <task>
  ```


### ADE20K

| Method | Segmentaion Backbone | Classification Backbone | Segmentation Crop Size |  PQ   | AP   | mIoU (s.s) | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---:    |  :---:    | :---: | :---:| :---:           | :---:               | :---:   |  :---: |    :---:   |
| OneFormer+ViT-P | DiNAT-L |DinoV2-B | 1280&times;1280 | 51.9 | 37.8 | 58.6 | 59.0 | 309M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_ADE20k_base774_250point.pth) |
| OneFormer(COCO-Pretrained)+ViT-P | DiNAT-L |DinoV2-B | 1280&times;1280 | 54.0 | 40.7 | 59.7 | 59.9 | 309M | [config](configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_ADE20k_base774_250point.pth) |
| OneFormer(emb_dim=256)+ViT-P | InternImage-H |DinoV2-L | 896&times;896 | 54.5 | 40.6 | 61.2 | 61.6 | 1.4B | [config](configs/ade20k/intern_image/oneformer_intern_image_huge_bs16_160k_896x896.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_intern_image_h_oneformer_ade20k_160k.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_ADE20k_798_250point.pth) |


### Cityscapes

| Method | Segmentaion Backbone	 | Classification Backbone|  PQ   | AP   | mIoU (s.s) | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    |:---:    | :---: | :---:| :---:      | :---:          | :---:   |  :---: |    :---:   |
| OneFormer(Mapillary Vistas-Pretrained)+ViT-P | ConvNeXt-L |DinoV2-B | 70.1 | 49.0 | 84.9 | 85.5 | 306M | [config](configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_large_bs16_90k.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_l_oneformer_cityscapes_90k.pth) &#124; [classification model]() |
| OneFormer(emb_dim=256)+ViT-P | InternImage-H |DinoV2-L| 70.8 | 50.6 | 85.4 | 85.9 | 1.4B | [config](configs/cityscapes/intern_image/oneformer_intern_image_huge_bs16_90k.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_intern_image_h_oneformer_cityscapes_90k.pth) &#124; [classification model]() |

### COCO

| Method | Segmentaion Backbone | Classification Backbone	|  PQ    | AP | mIoU | #params | config | Checkpoint |
|   :---:| :---:    |  :---: | :---: | :---:              | :---:   | :---:  |  :---: |    :---:   |
| OneFormer+ViT-P	 | DiNAT-L |DinoV2-B | 58.0 | 49.5 | 68.6 | 309M | [config](configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml) | [segmentation model](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) &#124; [classification model](https://huggingface.co/Sajjad-Sh33/ViT-P/resolve/main/model_coco_base793_150points.pth) |

