#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint="gpu_22g+"
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/1/user/sroy/class-inc/logs/out.stdout
#SBATCH --error=/scratch/1/user/sroy/class-inc/logs/error.stderr

# the bash script must be run from the root folder of this project

# The bash script takes the following positional arguments
# $1 gpu id. e.g., 0
# $2 task name. e.g., voc, voc-5, voc-2
# $3 name of the exp. It must match with the training folder name. e.g. RaSP
# $4 the step at which you would like to make an inference. e.g. t=3
# $5 semantic prior loss. 1 for RaSP and 0 for Wilson

# some examples:
# bash run/eval/eval-coco-to-voc.sh 0 voc-5 wilson 2 0
# bash run/eval/eval-coco-to-voc.sh 0 voc rasp 1 1

# for running on slurm:
# sbatch run/eval/eval-coco-to-voc.sh 0 voc-2 rasp 7 1

conda activate /home/sroy/workspace/environments/wilson/

export CUDA_VISIBLE_DEVICES=$1

port=$(python get_free_port.py)
echo ${port}

# make it anonymous if the code has to be submitted anonymously
logdir=/scratch/1/user/sroy/class-inc/logs
ckptdir=/scratch/1/user/sroy/class-inc/checkpoints
ext_dir=/scratch/1/user/sroy/ciss/ilsvrc2012

alias exp='python -m torch.distributed.launch --nproc_per_node=1 --master_port ${port} eval.py --num_workers 4 --sample_num 8 --logdir ${logdir} --ckpt_root ${ckptdir}'
shopt -s expand_aliases

task=$2
exp_name=$3
step=$4
sem_sim=$5

dataset=coco-voc
batch_size=1 # set batch_size=1 if --visualize_images is set, otherwise visualization code will break.

path=${ckptdir}/step/${dataset}-${task}/
echo ${path}

## semantic similarity
if [ ${sem_sim} -eq 1 ]
then
  ss="--semantic_similarity"
else
  ss=""
fi
echo ${ss}

dataset_pars="--dataset ${dataset} --task ${task} --batch_size ${batch_size}"
pretr_FT=${path}FTwide_bce_0.pth

if [ ${step} -eq 1 ]; then
  prev_ckpt_path=$pretr_FT
else
  prev_ckpt_path=${path}${exp_name}_$((${step}-1)).pth
fi
curr_ckpt_path=${path}${exp_name}_${step}.pth

exp --name ${exp_name} --step ${step} --weakly ${dataset_pars} --step_ckpt $prev_ckpt_path --curr_step_ckpt ${curr_ckpt_path} \
  --sample_num 32 --cam 'ngwp' --affinity --visualize_images ${ss}

echo "I am finishing"