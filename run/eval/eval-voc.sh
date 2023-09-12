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
# $2 overlapped setting indicator. 1 for overlapped and 0 for disjoint
# $3 task name. e.g., 15-5, 10-1, etc
# $4 name of the exp. It must match with the training folder name. e.g. RaSP-memory
# $5 the step at which you would like to make an inference. e.g. t=3
# $6 semantic prior loss. 1 for RaSP and 0 for Wilson

# some examples:
# bash run/eval/eval-voc.sh 0 0 15-5 wilson 1 0
# bash run/eval/eval-voc.sh 0 1 10-5-5 rasp 2 1

# for running on slurm:
# sbatch run/eval/eval-voc.sh 0 0 10-2 rasp 4 1

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

overlap=$2
task=$3
exp_name=$4
step=$5
sem_sim=$6

dataset=voc
batch_size=1 # set batch_size=1 if --visualize_images is set, otherwise visualization code will break.

if [ ${overlap} -eq 0 ]; then
  path=${ckptdir}/step/${dataset}-${task}/
  ov=""
else
  path=${ckptdir}/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
fi
echo ${path}

## semantic similarity
if [ ${sem_sim} -eq 0 ]
then
  ss=""
else
  ss="--semantic_similarity"
fi
echo ${ss}

dataset_pars="--dataset ${dataset} --task ${task} --batch_size ${batch_size} $ov"
pretr_FT=${path}FT_bce_0.pth

if [ ${step} -eq 1 ]; then
  prev_ckpt_path=$pretr_FT
else
  prev_ckpt_path=${path}${exp_name}_$((${step}-1)).pth
fi
curr_ckpt_path=${path}${exp_name}_${step}.pth

exp --name ${exp_name} --step ${step} --weakly ${dataset_pars} --step_ckpt $prev_ckpt_path --curr_step_ckpt ${curr_ckpt_path} \
  --sample_num 32 --cam 'ngwp' --affinity --visualize_images ${ss}

echo "I am finishing"