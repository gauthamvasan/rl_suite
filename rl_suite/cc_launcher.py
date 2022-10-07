import os
import time
from tqdm import tqdm

visual_steps = {
        "ball_in_cup": 200000,
        "dm_reacher_hard": 200000,
        "dm_reacher_easy": 200000,
        }

non_visual_steps = {
        "ball_in_cup": 200000,
        "dm_reacher_hard": 200000,
        "dm_reacher_easy": 200000,
        }

def generate_exps():
    exps = []
    
    algos = ["sac", "sac_rad"]
    algos = ["sac"]
    envs = ["ball_in_cup", "dm_reacher_hard", "dm_reacher_easy"]
    envs = ["dm_reacher_easy"]
    timeouts = [50, 100, 500, 1000]
    seeds = range(30)
    for algo in algos:
        description = "2nd_paper_with_image" if algo == "sac_rad" else "2nd_paper_no_image"
        for timeout in timeouts:
            for env in envs:
                N = visual_steps[env] if algo == "sac_rad" else non_visual_steps[env]
                init_steps = N//100
                for seed in seeds:
                    exp = {
                        "env": env,
                        "seed": str(seed),
                        "N": str(N),
                        "timeout": str(timeout),
                        "algo": algo,
                        "replay_buffer_capacity": "100000",
                        "init_steps": str(init_steps),
                        "description": description,
                        "output_filename": f'{algo}_{env}_timeout={timeout}_seed={seed}_{description}_%j.out',
                    }
                    exps.append(exp)

    return exps

def run_exp():
    exps = generate_exps()

    for exp in tqdm(exps):
        env = exp["env"]
        seed = exp["seed"]
        N = exp["N"]
        timeout = exp["timeout"]
        algo = exp["algo"]
        replay_buffer_capacity = exp["replay_buffer_capacity"]
        init_steps = exp["init_steps"]
        description = exp['description']
        output_filename = exp["output_filename"]

        requested_time = '00:45:00' if algo == "sac" else '05:00:00'
        requested_mem = '4G' if algo == "sac" else '12G'
        params = [  
            'sbatch',
            '--time='+requested_time,
            '--mem='+requested_mem,
            "--output="+output_filename,
            'cc_job.sh', 
            env,
            seed, 
            N,
            timeout,
            algo,
            replay_buffer_capacity,
            init_steps,
            description,
        ]
        command = " ".join(params)
        os.system(command)
        time.sleep(1)
        
if __name__ == '__main__':
    run_exp()
