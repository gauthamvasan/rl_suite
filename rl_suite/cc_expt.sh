#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-30 

module load cuda/11.2.2
source /home/vasan/src/rtrl/bin/activate

SECONDS=0
python learn_to_reset_time.py --description "reward_neg1_g995_rt09" --init_steps 5000 --work_dir "/home/vasan/src/rl_suite/rl_suite/results/ball_in_cup_baseline/reset_time_in_obs" --N 151000 --gamma 0.995 --env "ball_in_cup" --penalty 1 --timeout 500 --actor_hidden_sizes "512 512" --critic_hidden_sizes "2048 2048" --l2_reg 0.0001 --reset_thresh 0.9 --seed $SLURM_ARRAY_TASK_ID
echo "Baseline job $seed took $SECONDS"
