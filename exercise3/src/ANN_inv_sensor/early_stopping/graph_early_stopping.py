#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import linregress

xs = []
ys = []
zs = []

if len(sys.argv) < 2:
    print('usage: ./graph_early_stopping.py dir1 [dir2 [...]]')
    sys.exit(1)

outdirs = sys.argv[1:]
print('loading data from dirs: {} (pass as arguments to this script)'.format(outdirs))

for outdir in outdirs:
    for filename in os.listdir(outdir):
        with open(os.path.join(outdir, filename), 'r') as f:
            job = json.loads(f.read())
            # ignore partially completed jobs
            if 'error' in job.keys():
                xs.append(job['epochs'])
                ys.append(job['error'])
                zs.append(job['this_job_time'])

def transpose_list(l):
    return np.asarray(l).T.tolist()

def sort_by_first(*cols):
    num_rows = len(cols[0])
    data = transpose_list(cols) # transpose to get list of rows
    data = sorted(data, key=lambda elem: elem[0]) # sort based on first column
    return transpose_list(data)

xs, ys, zs = sort_by_first(xs, ys, zs)

slope, intercept, r, p, stderr = linregress(xs, zs)
print('time = {}*x + {} with PMCC={}'.format(slope, intercept, r))

plt.figure(1)
plt.plot(xs, ys)
plt.ylabel('validation error')
plt.xlabel('epochs')

plt.figure(2)
plt.plot(xs, zs)
reg = intercept + slope*np.array(xs, dtype=float)
plt.plot(xs, reg)
plt.ylabel('time taken')
plt.xlabel('epochs')

plt.show()



