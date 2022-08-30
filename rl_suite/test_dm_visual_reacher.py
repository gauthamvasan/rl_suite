import subprocess
from multiprocessing import Pool

def generate_exps():
    exps = []
    
    seeds = range(30)
    
    for seed in seeds:
        exp = {
            "name": f'dm_visual_reacher_hard_seed={seed}_with_image',
            "env": "dm_reacher_hard",
            "seed": seed,
            "N": 350000
        }
        exps.append(exp)

    return exps

exps = generate_exps()

def run_exp(exp):
    env = exp["env"]
    seed = exp["seed"]
    N = exp["N"]
    with open(exp["name"]+'.out', 'w') as out_file:
        param = ['python3', '-u', 'task_dm_visual_reacher.py', 
                                    '--env', env,
                                    '--seed', str(seed), 
                                    '--N', str(N),
                ]

        subprocess.run(param, stdout=out_file)

if __name__ == '__main__':
    workers = 5
    with Pool(processes=workers) as p:
        p.map(run_exp, exps)