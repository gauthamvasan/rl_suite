# rl_suite
A simple, easy-to-use testbed for Reinforcement Learning. This supports multiple RL benchmark tasks in both
guiding rewards and minimum-time formulation.

## Implemented algorithms
- SAC
- PPO
- SAC + RAD
- Asynchronous SAC

## How to use?
```bash
python sac_experiment.py --env "dm_reacher_easy" --seed 42 --N 201000 --timeout 100 --algo "sac" --replay_buffer_capacity 100000 --results_dir "./results" --init_steps 1000
```

## Compute Canada
Example `cc_expt.sh` script for running parellel SAC experiments on a node with 1 GPU and multiple cores.

```bash
#!/bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1500M
#SBATCH --gres=gpu:1

source /home/vasan/src/rtrl/bin/activate

timeout=50
env="dm_reacher_easy"

parallel -j 15 python sac_experiment.py --env $env --timeout $timeout --N 201000 --algo "sac" --replay_buffer_capacity 100000 --results_dir "/home/vasan/scratch/min_time_paper/$env" --init_steps 20000 ::: --seed ::: {1..15}
```