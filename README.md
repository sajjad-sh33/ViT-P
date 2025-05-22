
# The Missing Point in Vision Transformers for Universal Image Segmentation

This repository includes the implementation for our paper, The Missing Point in Vision Transformers for Universal Image Segmentation.

<img src="images/SOTA.png" width="100%"/>

ViT-P is a novel two-stage segmentation framework that decouples mask generation from classification by first creating robust, class-agnostic mask proposals and then refining them with a point-based Vision Transformer. It operates without the need for pre-training, seamlessly integrating existing transformer backbones into diverse dense prediction tasks. By leveraging an innovative annotation strategy that incorporates both coarse and bounding box labels. Extensive experiments on COCO, ADE20K, and Cityscapes validate ViT-P’s state-of-the-art performance across panoptic, instance, and semantic segmentation tasks.

<img src="images/Model_Architecture.png" width="100%"/>


## Evaluation

For evaluation, please refer to the **Oneformer** and **Internimage** folders where you can find detailed scripts and instructions:  
- [Oneformer+ViT-P Evaluation](./OneFormer)  
- [Internimage+ViT-P Evaluation](./InternImage)




## Training

Our training code is built based on the [DinoV2 repository](https://github.com/facebookresearch/dino-v2). 

To train the `ViT-P` on SLURM cluster, run:

```bash
sh train_ADE20K_base.sh
```



### Datasets
To create the training datasets, click on the [training dataset preparation](./ViT-P/datasets/README.md) script provided within the repository. Detailed instructions are available in the corresponding file.


### Installation
The training code requires PyTorch (version 2.0 or higher) and xFormers (version 0.0.18 or higher). You may work with any torch and xFormers versions that are mutually compatible.

To install the training dependencies, run:
```bash
pip install -r train_requirements.txt
```
**Note:** Additional installation steps are required for evaluation; please check the guidelines in both the **Internimage** and **Oneformer** folders.

- To execute the commands provided in the next section for evaluation, the dinov2 package must be included in the Python module search path.

```bash
cd ./ViT-P
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd ../OneFormer
  ```





