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
python learn_to_reset.py --description "g99_rt09" --reset_thresh 0.9 --work_dir "/home/vasan/scratch/reset_action/dm_reacher_easy" --N 501000 --env "dm_reacher_easy" --seed $SLURM_ARRAY_TASK_ID --penalty -1
echo "Baseline job $seed took $SECONDS"
