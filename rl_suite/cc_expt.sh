#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-10:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-30 

module load cuda/11.2.2
source /home/vasan/src/rtrl/bin/activate

SECONDS=0
python learn_to_reset.py --description "reset_thresh_09_g995" --init_steps 5000 --work_dir "./results/reset_action/dot_reacher_hard" --N 300000 --gamma 0.995 --seed $SLURM_ARRAY_TASK_ID --reset_thresh 0.9 --timeout 20000 --pos_tol 0.1 --vel_tol 0.05
echo "Baseline job $seed took $SECONDS"
