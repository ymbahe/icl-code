"""
Small helper program to combine origin tables produced by different MPI tasks.

Started 04-Jul-2019

Updated 21-Jun-2023
"""

import h5py as h5
import numpy as np
import os
from shutil import copyfile
from pdb import set_trace

# Combine entries across all simulations?
combine_sims = True

# Does the input catalogue contain parents?
with_parents = False

# File (prefix) of the files to combine
dataloc = 'OriginMasses_Test'

# Number of files to combine
nfiles = 1

# --------------------------------------------------------------------------

outloc = dataloc + '.hdf5'

data = {}
hdr_attrs = {}
with h5.File(dataloc + '.0.hdf5', 'r') as f:
    dsets = list(f.keys())
    if 'Header' in dsets: dsets.remove('Header')
    for key in dsets:
        data[key] = f[key][...]
    for key in f['Header'].attrs:
        hdr_attrs[key] = f['Header'].attrs[key]

# Now copy in the content of the other files
for ifile in range(1, nfiles):
    with h5.File(dataloc + f'.{ifile}.hdf5', 'r') as f:
        for dset in dsets:
            data[dset] += f[dset][...]

# Done reading and stacking all the data. If desired, collapse output array
# over the simulation axis (0), since it is redundant.
if combine_sims:
    for dset in dsets:
        data[dset] = np.sum(data[dset], axis=0)

# Write output
with h5.File(dataloc + '.hdf5', 'w') as f:
    for dset in dsets:
        f[dset] = data[dset]
    for key in hdr_attrs:
        f['Header'].attrs[key] = hdr_attrs[key]


print("Done!")
