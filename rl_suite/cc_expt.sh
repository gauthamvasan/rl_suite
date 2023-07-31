#!/bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=3072M
#SBATCH --gres=gpu:1

module load mujoco/2.3.6
source /home/vasan/src/rtrl/bin/activate

reward_types=("dense" "sparse")
maze_types-("open" "T" "plus")
timeouts=(1000 5000)

parallel -j 15 python sac_experiment.py --env "point_maze" --N 201000 --algo "sac" --replay_buffer_capacity 100000 --results_dir "./results" --init_steps 20000 ::: --seed ::: {1..15} ::: --maze_type ::: ${maze_types[@]} ::: --reward_type ::: ${reward_types[@]} ::: --timeout ::: ${timeouts[@]}
