import json
import sys

total = 0
for p in sys.argv[1:]:
    with open(p, 'r') as f:
        logs = json.load(f)
        times = logs['epoch_time']
        total += sum(times)
print(total)