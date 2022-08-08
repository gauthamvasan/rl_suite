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
python sac_experiment.py --description "reward_neg01_g1" --init_steps 5000 --work_dir "/home/vasan/src/rl_suite/rl_suite/results/ball_in_cup/reward_neg01_t1000" --N 150000 --gamma 1 --env "ball_in_cup" --penalty 0.1 --timeout 1000 --actor_hidden_sizes "1024 1024" --critic_hidden_sizes "1024 1024" --seed $SLURM_ARRAY_TASK_ID
echo "Baseline job $seed took $SECONDS"
