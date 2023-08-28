# Setting up env on Compute Canada

## Virtualenv setup
```bash
module load gcc opencv cuda/11.8 python/3.10 mujoco/2.3.6
virtualenv --no-download --clear ~/RL && source ~/RL/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --no-index gymnasium_robotics gymnasium gym
pip install dm_control==1.0.13
pip install beautifultable matplotlib tqdm
pip install --no-index pygame
```

### Comments
- Use local packages like `rl_suite` and `incremental_rl` using PYTHONPATH

## Example script 
```bash
#!/bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00
#SBATCH --mem-per-cpu=3072M
#SBATCH --gres=gpu:1


module load gcc opencv cuda/11.8 mujoco/2.3.6 python/3.10
source /home/vasan/RL/bin/activate

export PYTHONPATH="$PYTHONPATH:/home/vasan/src/rl_suite:/home/vasan/src/incremental_rl"

maze_types=("open" "T" "plus")
parallel -j 15 python sac_experiment.py --env "point_maze" --N 201000 --timeout 1000 --reward_type "sparse" --algo "sac" --replay_buffer_capacity 100000 --results_dir "/home/vasan/scratch/tro_paper" --init_steps 20000 ::: --seed ::: {1..30} ::: --maze_type ::: ${maze_types[@]}
```

### Another example
```bash
#!/bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00
#SBATCH --mem=75000M
#SBATCH --gres=gpu:1

module load gcc opencv cuda/11.8 python/3.10 mujoco/2.3.6
source /home/vasan/RL/bin/activate
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="$PYTHONPATH:/home/vasan/src/rl_suite:/home/vasan/src/incremental_rl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

envs=("Hopper-v4" "Humanoid-v4" "Ant-v4")

parallel -j 30 python sac_experiment.py --N 1001000 --algo "sac" --replay_buffer_capacity 1000000 --results_dir "/home/vasan/scratch/sac_baseline" --init_steps 1000 --timeout 1000 --actor_hidden_sizes "256,256" --critic_hidden_sizes "256,256" --update_every 1 --reset_penalty_steps 0 --bootstrap_terminal 1 ::: --env ::: ${envs[@]} ::: --seed ::: {1..10}
```