import argparse
import os
import time
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

visual_steps = {
        "ball_in_cup": 200000,
        "dm_reacher_hard": 200000,
        "dm_reacher_easy": 200000,
        "mj_reacher": 200000
        }

non_visual_steps = {
        "ball_in_cup": 200000,
        "dm_reacher_hard": 200000,
        "dm_reacher_easy": 200000,
        "mj_reacher": 200000
        }

def generate_exps():
    exps = []
    
    algos = ["sac", "sac_rad"]
    algos = ["sac",]
    envs = ["ball_in_cup", "dm_reacher_hard", "dm_reacher_easy"]
    envs = ['mj_reacher']
    timeouts = [10, 25, 50, 100, 500, 1000, 5000]
    timeouts = [5000]
    seeds = range(1)
    factor = 10
    for algo in algos:
        description = "2nd_paper_with_image" if algo == "sac_rad" else "2nd_paper_no_image"
        for timeout in timeouts:
            for env in envs:
                N = visual_steps[env] if algo == "sac_rad" else non_visual_steps[env]
                init_steps = N//factor
                for seed in seeds:
                    exp_dir = env+('/visual' if algo == "sac_rad" else "/non_visual")+f"/timeout={timeout}/seed={seed}/"
                    exp = {
                        "env": env,
                        "seed": str(seed),
                        "N": str(N),
                        "timeout": str(timeout),
                        "algo": algo,
                        "replay_buffer_capacity": "100000",
                        "init_steps": str(init_steps),
                        "experiment_dir": exp_dir,
                        "description": description,
                        "output_filename": f'{env}_timeout={timeout}_seed={seed}_{description}',
                    }
                    exps.append(exp)

    return exps

def cc_exp(exps):
    for exp in tqdm(exps):
        env = exp["env"]
        seed = exp["seed"]
        N = exp["N"]
        timeout = exp["timeout"]
        algo = exp["algo"]
        replay_buffer_capacity = exp["replay_buffer_capacity"]
        init_steps = exp["init_steps"]
        experiment_dir = exp["experiment_dir"]
        description = exp['description']
        output_filename = exp["output_filename"]

        requested_time = '00:35:00' if algo == "sac" else '06:00:00'
        requested_mem = '3G' if algo == "sac" else '12G'
        params = [  
            'sbatch',
            '--time='+requested_time,
            '--mem='+requested_mem,
            "--output="+output_filename+'_%j.out',
            './cc_job.sh', 
            env,
            seed, 
            N,
            timeout,
            algo,
            replay_buffer_capacity,
            init_steps,
            experiment_dir,
            description,
        ]

        command = " ".join(params)
        os.system(command)
        time.sleep(1)

def workstation_exp(exp):
    env = exp["env"]
    seed = exp["seed"]
    N = exp["N"]
    timeout = exp["timeout"]
    algo = exp["algo"]
    replay_buffer_capacity = exp["replay_buffer_capacity"]
    init_steps = exp["init_steps"]
    experiment_dir = exp["experiment_dir"]
    description = exp['description']
    output_filename = exp["output_filename"]

    with open(output_filename+'.out', 'w') as out_file:
        param = ['python3', '-u', 'sac_experiment.py', 
                                    '--env', env,
                                    '--seed', seed, 
                                    '--N', N,
                                    '--timeout', timeout,
                                    '--algo', algo,
                                    '--replay_buffer_capacity', replay_buffer_capacity,
                                    '--init_steps', init_steps,
                                    '--experiment_dir', experiment_dir,
                                    '--description', description
                ]
        subprocess.run(param, stdout=out_file, stderr=out_file)

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--target', '-t', required=True, type=str)
        args = parser.parse_args()
        
        return args

def run_exp():
    exps = generate_exps()
    args = parse_args()
    
    if args.target == 'cc':
        cc_exp(exps)
    elif args.target == 'workstation':
        workers = 5
        with Pool(processes=workers) as p:
            p.map(workstation_exp, exps)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    work_dir = Path(__file__).parent
    os.chdir(work_dir)
    run_exp()
