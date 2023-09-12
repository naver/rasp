#!/bin/bash

#SBATCH --job-name=rasp
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
# $2 overlapped setting indicator. 1 for overlapped and 0 for disjoint
# $3 task name. e.g., 15-5, 10-1, etc
# $4 name of the exp. e.g. RaSP
# $5 number of WS incremental steps. e.g., 1 for task 15-5, 5 for task 15-1, etc

# some examples:
# bash run/run-voc-rasp.sh 0,1 0 15-5 RaSP 1
# bash run/run-voc-rasp.sh 0,1 1 10-5-5 RaSP 2

# for running on slurm:
# sbatch run/run-voc-rasp.sh 0,1 0 10-2 RaSP 5

conda activate /home/sroy/workspace/environments/wilson/

export CUDA_VISIBLE_DEVICES=$1

port=$(python get_free_port.py)
echo ${port}

logdir=/scratch/1/user/sroy/class-inc/logs
ckptdir=/scratch/1/user/sroy/class-inc/checkpoints
ext_dir=/scratch/1/user/sroy/ciss/ilsvrc2012

alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 4 --sample_num 8 --logdir ${logdir} --ckpt_root ${ckptdir}'
shopt -s expand_aliases

overlap=$2
task=$3
exp_name=$4
nb_incremental_steps=$5

dataset=voc
epochs=40
lr_init=0.01
lr=0.001
batch_size=24

if [ ${overlap} -eq 0 ]; then
  path=${ckptdir}/step/${dataset}-${task}/
  ov=""
else
  path=${ckptdir}/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
fi
echo ${path}

dataset_pars="--dataset ${dataset} --task ${task} --batch_size ${batch_size} --epochs ${epochs} $ov"
pretr_FT=${path}FT_bce_0.pth

if [[ ! -f $pretr_FT ]]
then
  exp --name FT_bce --step 0 --bce --lr ${lr_init} ${dataset_pars}
fi

for i in $(seq $nb_incremental_steps)
do
  if [ ${i} -eq 1 ]; then
    ckpt_path=$pretr_FT
  else
    ckpt_path=${path}${exp_name}_$((i-1)).pth
  fi

  exp --name ${exp_name} --step ${i} --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $ckpt_path \
    --loss_de 1 --lr_policy warmup --sample_num 32 --cam 'ngwp' --affinity \
    --val_interval 5 --pseudo_ep 5 \
    --semantic_similarity
done

echo "I am finishing"