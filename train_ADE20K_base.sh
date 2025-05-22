#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00-07:00:00


cp -r dinov2 $SLURM_TMPDIR
unzip -q ./backup/ADEChallengeData2016/ADEChallengeData2016.zip -d $SLURM_TMPDIR/ViT-P/datasets
echo "*****Images are ready*****"


module purge
module load gcc arrow
module spider opencv
module load opencv/4.9.0
source ~/py311/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/projects/ViT-P

cd $SLURM_TMPDIR
cd ViT-P

srun --unbuffered python dinov2/train/train.py --config-file dinov2/configs/train/vitb14_ADE20k.yaml --output-dir ./OUTPUT_DIR --no-resume

tar -cf ~/projects/model_ADE20k_base.tar checkpoint.pth
