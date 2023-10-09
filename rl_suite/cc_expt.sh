#!/bin/bash
#SBATCH --account=def-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1-00:00
#SBATCH --mem=150000M
#SBATCH --gres=gpu:1
#SBATCH --array=1-15

module load gcc opencv cuda/11.8 mujoco/2.3.6 python/3.10
source /home/vasan/RL/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/vasan/src/rl_suite:/home/vasan/src/incremental_rl"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

maze_types=("min_time")
reward_types=("sparse" "dense")
SLURM_ARRAY_TASK_ID=13
B=($((SLURM_ARRAY_TASK_ID + 15)))

for maze_type in ${maze_types[@]}
do
    for reward_type in ${reward_types[@]}
    do
        # for seed in $(seq $SLURM_ARRAY_TASK_ID $B);
        for seed in $SLURM_ARRAY_TASK_ID $B
        do
            SECONDS=0
            python sac_experiment.py --env "point_maze" --seed $seed --N 501000 --timeout 500 --gamma 0.995 --reward -1 --algo "sac_rad" --replay_buffer_capacity 100000 --results_dir "/home/vasan/scratch/tro_paper" --init_steps 20000 --cnn_architecture "V2" --use_image --device "cuda" --maze_type $maze_type --reward_type $reward_type &
            sleep 60
            echo "$SECONDS s elapsed."
        done
    done
done
wait