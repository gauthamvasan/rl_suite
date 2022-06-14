#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-10%5

module load python/3.9
module load cuda/11.2.2
source /home/vasan/src/rtrl/bin/activate

env="sparse_reacher"
SECONDS=0
python sac_experiment.py --env $env --gamma 1 --description "test" --seed $SLURM_ARRAY_TASK_ID --timeout 500 --init_steps 1000 --actor_hidden_sizes "1024 1024" --critic_hidden_sizes "1024 1024" --work_dir "/home/vasan/src/rl_suite/rl_suite/results/$env"
echo "Baseline job $seed took $SECONDS"