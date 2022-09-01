#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-30 

module load cuda/11.2.2
source /home/vasan/src/rtrl/bin/activate

SECONDS=0
python sac_experiment.py --description "baseline_t100" --timeout 100 --algo "sac_rad" --work_dir "/home/vasan/scratch/reset_action/visual_dot_reacher/sac" --N 251000 --env "visual_dot_reacher" --seed $SLURM_ARRAY_TASK_ID --penalty -1
echo "Baseline job $seed took $SECONDS"
