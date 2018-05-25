#!/usr/bin/env python
import pdb

import sys
import os
import re

import cPickle

save_files = os.listdir('MC_runs')

metadata = []
for mc_file in save_files:
    if re.search('n=350', mc_file) and os.path.splitext(mc_file)[1] == '.p':
        with open('MC_runs/{}'.format(mc_file), 'rb') as pfile:
            save_data = cPickle.load(pfile)
        this_metadata = (
            mc_file,
            save_data[0]['pilot']['P_landout_acceptable'],
            save_data[0]['pilot']['gear_shifting'],
            save_data[0]['thermal_field']._thermals[0].P_work)
        metadata.append(this_metadata)

with open('mc_metadata.txt', 'w') as metadata_file:
    for entry in metadata:
        metadata_file.write(
            '{}\n\tRisk_tolerance: {}, gear_shifting: {}, P_work: {}\n'.format(
                entry[0], entry[1], entry[2], entry[3]))
