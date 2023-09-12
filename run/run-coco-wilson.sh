#!/bin/bash

#SBATCH --job-name=wilson
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64gb
#SBATCH --constraint="gpu_22g+"
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/1/user/sroy/class-inc/logs/out.stdout
#SBATCH --error=/scratch/1/user/sroy/class-inc/logs/error.stderr

# the bash script must be run from the root folder of this project

# The bash script takes the following positional arguments
# $1 gpu ids separated by commas. e.g., 0,1 for running on 2 gpus
# $2 task name. e.g., voc, voc-5, voc-2
# $3 name of the exp. e.g. WILSON
# $4 number of WS incremental steps. e.g., 1 for task voc, 4 for task voc-5, etc

# some examples:
# bash run/run-coco-wilson.sh 0,1 voc wilson 1
# bash run/run-coco-wilson.sh 0,1 voc-5 wilson 4

# for running on slurm:
# sbatch run/run-coco-wilson.sh 0,1 voc-2 wilson 10

conda activate /home/sroy/workspace/environments/wilson/

export CUDA_VISIBLE_DEVICES=$1

logdir=/scratch/1/user/sroy/class-inc/logs
ckptdir=/scratch/1/user/sroy/class-inc/checkpoints

port=$(python get_free_port.py)
echo ${port}

alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 4 --sample_num 8 --logdir ${logdir} --ckpt_root ${ckptdir}'
shopt -s expand_aliases

task=$2 # e.g. voc, voc-5
exp_name=$3
nb_incremental_steps=$4

dataset=coco-voc
epochs=30
lr_init=0.01
lr=0.001
bs=24

path=${ckptdir}/step/${dataset}-${task}/
dataset_pars="--dataset ${dataset} --task ${task} --batch_size ${bs} --epochs ${epochs} $ov --val_interval 2"
pretr_FT=${path}FTwide_bce_0.pth

if [[ ! -f $pretr_FT ]]
then
    exp --name FTwide_bce --step 0 --lr ${lr_init} ${dataset_pars} --bce
fi

for i in $(seq $nb_incremental_steps)
do
    if [ ${i} -eq 1 ]; then
        ckpt_path=$pretr_FT
    else
        ckpt_path=${path}${exp_name}_$((i-1)).pth
    fi

    exp --name ${exp_name} --step ${i} --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $ckpt_path \
        --loss_de 1 --lr_policy warmup --affinity
done

echo "I am finishing"
