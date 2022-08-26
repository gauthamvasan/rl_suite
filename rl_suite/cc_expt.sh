#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-30 

module load cuda/11.2.2
source /home/vasan/src/rtrl/bin/activate

SECONDS=0
python sac_experiment.py --description "g99_t500" --init_steps 5000 --work_dir "/home/vasan/scratch/reset_action/ball_in_cup/sac" --N 251000 --gamma 0.99 --env "ball_in_cup" --seed --init_steps 5000 --update_every 2 --update_epochs 1 --batch_size 256 --actor_lr 0.0003 --critic_lr 0.0003 --alpha_lr 0.0003 --actor_update_freq 1 --critic_target_update_freq 1 --init_temperature 0.1 --critic_tau 0.005 --encoder_tau 0.005 --l2_reg 0.0001 --actor_hidden_sizes "512 512" --critic_hidden_sizes "512 512" --nn_activation "relu" --rad_offset 0.01 --seed $SLURM_ARRAY_TASK_ID
echo "Baseline job $seed took $SECONDS"
