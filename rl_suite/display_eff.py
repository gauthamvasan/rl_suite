import subprocess
from pathlib import Path
import re
import time, datetime

envs = ["ball_in_cup", "dm_reacher_easy", "dm_reacher_hard"]
folder = Path(__file__).parent
times = {
    "ball_in_cup": 0,
    "dm_reacher_easy": 0,
    "dm_reacher_hard": 0,
}
counts = {
    "ball_in_cup": 0,
    "dm_reacher_easy": 0,
    "dm_reacher_hard": 0,
}
largest_seconds = -1
longest_env = None
longest_job_id = None
for out_file in folder.iterdir():
    if out_file.name.endswith(".out"):
        pattern = f"({'|'.join(envs)}).*_" + r"(\d+).out"
        match = re.match(pattern, out_file.name)
        env = match.group(1)
        run_id = match.group(2)
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
            print(f"{out_file.name}: {res[3]}")

print()
for (env, total_seconds) in times.items():
    if counts[env] != 0:
        avg_seconds = total_seconds / counts[env]
        avg_time_string = str(datetime.timedelta(seconds=avg_seconds))
        print(f"{env}: {avg_time_string}, counts={counts[env]}")

longest_time_string = str(datetime.timedelta(seconds=largest_seconds))
print(f"longest env: {env}, longest time: {longest_time_string}, job_id: {longest_job_id}")
