"""
Find target galaxies at a specified snapshot.
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
import hydrangea_tools as ht
from pdb import set_trace
import os

basedir = '/virgo/simulations/Hydrangea/10r200/'

outloc = '/u/ybahe/ANALYSIS/IMAGES/cluster_script_s29_xt.sh'

f = open(outloc, 'a')
f.write('#! /bin/bash\n')
f.close()

isnap = 29

for isim in range(30):

    print("Processing simulation {:d}..." .format(isim))

    rundir = basedir + 'CE-{:d}/HYDRO/' .format(isim)


    if not os.path.isdir(rundir): continue

    subdir = st.form_files(rundir, isnap)

    hldir = rundir + 'highlev/'
    freyadir = ht.clone_dir(hldir)
    
    fgtloc = hldir + 'FullGalaxyTables.hdf5'
    cantorloc = hldir + 'CantorCatalogue.hdf5'

    fof_m200 = st.eagleread(subdir, 'FOF/Group_M_Mean200', astro = True)[0]
    fof_r200 = st.eagleread(subdir, 'FOF/Group_R_Mean200', astro = True)[0]
    sh_grp = np.abs(
        st.eagleread(subdir, 'Subhalo/GroupNumber', astro = False))-1
    shi = yb.read_hdf5(fgtloc, 'SHI')[:, isnap]
    m200 = np.log10(fof_m200[sh_grp[shi]])+10.0
    r200 = fof_r200[sh_grp[shi]]

    contFlag = yb.read_hdf5(fgtloc, 'ContFlag')[:, isnap]

    #m200 = yb.read_hdf5(fgtloc, 'M200')[:, isnap]
    cenGal = yb.read_hdf5(cantorloc, 'CenGalExtended')[:, isnap]
    
    ngal = len(cenGal)
    ind_gal = np.nonzero((shi >= 0) & (cenGal == np.arange(ngal)) & 
                         (m200 > 14.2) & (contFlag < 2))[0]
    
    print("Found {:d} galaxies in sim {:d}..." .format(len(ind_gal), isim))

    if len(ind_gal) == 0: continue

    f = open(outloc, 'a')

    for igal in ind_gal:
        f.write("nice /u/ybahe/anaconda3/bin/python3 "
                "cluster_image.py {:d} {:d} {:.5f} {:.5f}\n" 
                .format(isim, igal, m200[igal], r200[igal]))

    f.close()

