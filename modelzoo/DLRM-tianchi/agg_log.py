import re

PATTERN=re.compile(r"steps = (\d+) \((\d+\.\d+) sec\)")

def agg_log(log_file_path):
    with open(log_file_path, "r") as f:
        lines = f.readlines()
    steps = 0
    times = 0.0
    for line in lines:
        match = PATTERN.search(line)
        if match:
            steps = int(match.group(1))
            times += float(match.group(2))
    print(f"steps = {steps}, time = {times} sec, throughput = {steps/times} steps/sec")