#!/usr/bin/env python
import pdb

import os
import re
import copy
import datetime
import hashlib

import numpy

import cPickle

base_dir = 'MC_runs'

files = os.listdir(base_dir)
search_files = copy.deepcopy(files)

for f in files:
    load_files = [f,]
    if os.path.splitext(f)[1] != '.p':
        continue
    if not re.search('n=50', f):
        continue

    P_work = re.search('(Pwork=0\.[0-9]+)', f).groups()[0]
    P_accept = re.search('(Paccept=0\.[0-9]+)', f).groups()[0]
    shift = re.search('(shift=(?:True|False))', f).groups()[0]

    search_files.pop(search_files.index(f))
    for ff in search_files:
        if P_work in ff and P_accept in ff and shift in ff:
            files.pop(files.index(ff))
            load_files.append(ff)

    n = 0
    save_data = []
    for ff in load_files:
        with open(os.path.join(base_dir, ff), 'rb') as pfile:
            save_data.append(cPickle.load(pfile))
        n += 50
        if n >= 350:
            break

    run_id = hashlib.md5(
        str(datetime.datetime.now()) + str(numpy.random.randn())).hexdigest()
    save_path = '/storage/home/jjb481/work/soaring_risk_management/MC_runs/'
    save_path += '{}_n={}'.format(run_id, n)
    save_path += '_{}_{}_{}.p'.format(P_work, P_accept, shift)
    print(save_path)
    with open(save_path, 'wb') as pfile:
        cPickle.dump(save_data, pfile)
