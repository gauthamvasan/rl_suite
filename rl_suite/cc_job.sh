#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Sequential code.
#SBATCH --account=rrg-ashique

module load cuda/11.2.2
source ~/rl/bin/activate

ENV=$1
SEED=$2
N=$3
TIME_OUT=$4
ALGO=$5
BUFFER_SIZE=$6
INIT_STEPS=$7
EXPERIMENT_DIR=$8
DESCRIPTION=$9

python -u sac_experiment.py \
            --env $ENV \
            --seed $SEED \
            --N $N \
            --timeout $TIME_OUT \
            --algo $ALGO \
            --replay_buffer_capacity $BUFFER_SIZE \
            --init_steps $INIT_STEPS \
            --experiment_dir $EXPERIMENT_DIR \
            --description $DESCRIPTION
