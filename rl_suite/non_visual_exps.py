import subprocess
from multiprocessing import Pool

steps = {
        "ball_in_cup": 300000,
        "dm_reacher_hard": 400000,
        "dm_reacher_easy": 300000
        }

def generate_exps():
    exps = []
    
    envs = ["dm_reacher_hard"]
    timeouts = [50, 100, 500, 1000, 5000]
    seeds = range(30)

    for timeout in timeouts:
        for env in envs:
            for seed in seeds:
                exp = {
                    "name": f'sac_{env}_timeout={timeout}_seed={seed}_2nd_paper_no_image',
                    "env": env,
                    "seed": seed,
                    "timeout": timeout,
                    "description": "2nd_paper_no_image"
                }
                exps.append(exp)

    return exps

exps = generate_exps()

def run_exp(exp):
    env = exp["env"]
    seed = exp["seed"]
    timeout = exp["timeout"]
    N = steps[env]
    description = exp['description']
    with open(exp["name"]+'.out', 'w') as out_file:
        param = ['python3', '-u', 'sac_experiment.py', 
                                    '--env', env,
                                    '--seed', str(seed), 
                                    '--N', str(N),
                                    '--timeout', str(timeout),
                                    '--algo', 'sac',
                                    '--replay_buffer_capacity', '1000000',
                                    '--init_steps', str(N//100),
                                    '--description', description
                ]

        subprocess.run(param, stdout=out_file)

if __name__ == '__main__':
    workers = 5
    with Pool(processes=workers) as p:
        p.map(run_exp, exps)