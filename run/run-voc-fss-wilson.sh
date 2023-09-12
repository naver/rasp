#!/bin/bash
# launches VOC Single-Stage FSS experiments

#SBATCH --job-name=wilson-fss
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
# $1 gpu id
# $2 task name. eg: 5-0/5-1/5-2/5-3
# $3 name of the exp. e.g. WILSON

# some examples:
# bash run/run-voc-fss-wilson.sh 0 5-0 wilson-fss
# bash run/run-voc-fss-wilson.sh 0 5-1 wilson-fss

# for running on slurm:
# sbatch run/run-voc-fss-wilson.sh 0 5-2 wilson-fss


export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}

logdir=/scratch/1/user/sroy/class-inc/logs
ckptdir=/scratch/1/user/sroy/class-inc/checkpoints

alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run_fss.py --num_workers 4 --logdir ${logdir} --ckpt_root ${ckptdir}"
shopt -s expand_aliases

task=$2
exp_name=$3

dataset=voc
ft_batch_size=24
ft_epochs=30
lr_init=0.01

path=${ckptdir}/step/${dataset}-${task}/
echo ${path}

dataset_pars="--dataset ${dataset} --task ${task} --batch_size ${ft_batch_size} --epochs ${ft_epochs}"
pretr_FT=${path}FT_bce_0.pth

if [[ ! -f $pretr_FT ]]
then
    exp --name FT_bce --step 0 --bce --lr ${lr_init} ${dataset_pars} --val_interval 5
fi

gen_par="--task ${task} --dataset ${dataset} --batch_size 10"
lr=0.001
iter=1000

for ns in 2 5; do  # shot 2/5 images
  for is in 0 1 2; do  # image sampling indices
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
      
      exp --name ${exp_name} --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_bce_0.pth \
        --weakly --alpha 0.5 --loss_de 1 --lr_policy warmup --cam 'ngwp' --affinity \
        --val_interval 10 --pseudo_ep 10

  done
done

echo "I am finishing."