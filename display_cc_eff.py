import subprocess
from pathlib import Path
import re
import os
import time, datetime

res_folder = Path(__file__).parent/'results'
envs = next(os.walk(res_folder))[1]
times = {env: 0 for env in envs}
counts = {env: 0 for env in envs}
largest_seconds = -1
longest_env = None
longest_job_id = None

for env in envs:
    tasks = next(os.walk(res_folder/env))[1]
    for task in tasks:
        timeouts = next(os.walk(res_folder/env/task))[1]
        for timeout in timeouts:
            seeds = timeouts = next(os.walk(res_folder/env/task/timeout))[1]
            for seed in seeds:
                outputs_folder = res_folder/env/task/timeout/seed/'outputs/'
                for out_fname in outputs_folder.iterdir():
                    if out_fname.name.endswith(".txt"):
                        pattern = r"output_(\d+).txt"
                        match = re.match(pattern, out_fname.name)
                        run_id = match.group(1)
                        res = subprocess.check_output(['seff', run_id]).decode().split('\n')
                
                        if res[3] == 'State: COMPLETED (exit code 0)': # only counts finished jobs
                            duration = (res[7].split())[3]
                            try:
                                x = time.strptime(duration,'%H:%M:%S')
                            except:
                                duration = (res[8].split())[3]
                                x = time.strptime(duration,'%H:%M:%S')
                                
                            total_seconds = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
                            times[env] += total_seconds
                            counts[env] += 1

                            if total_seconds > largest_seconds:
                                largest_seconds = total_seconds
                                longest_env = env
                                longest_job_id = run_id

                        else:
                            print(f"{env}, {task}, {timeout}, {seed}, {run_id}: {res[3]}")

print()
for (env, total_seconds) in times.items():
    if counts[env] != 0:
        avg_seconds = total_seconds / counts[env]
        avg_time_string = str(datetime.timedelta(seconds=avg_seconds))
        print(f"{env}: {avg_time_string}, counts={counts[env]}")

longest_time_string = str(datetime.timedelta(seconds=largest_seconds))
print(f"longest env: {env}, longest time: {longest_time_string}, job_id: {longest_job_id}")
