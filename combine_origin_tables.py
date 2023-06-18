"""
Small helper program to combine origin tables produced by different MPI tasks.

Started 04-Jul-2019
"""

import yb_utils as yb
import numpy as np
import os
from shutil import copyfile
from pdb import set_trace

combine_sims = True
with_parents = False

dataloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/' +
          #'OriginMasses_08Jul19_HostMhalo5_RootMsub24_HostRad7_firstroot')
          # 'OriginMasses_2Aug19_HostCat_RootXTopGal_HostRad22_lastroot_noxd')
          #'OriginMasses_1Aug19_HostCat_RootTopGal_HostRad22_lastroot_noxd')
          #'OriginMasses_2Aug19_HostCat_RootMstar3_HostRad22_lastroot_withForgetfulParents_noxd')
          #'OriginMasses_5Aug19_HostCat_RootMstar3_HostRad7_firstroot_XD')
          #'OriginMasses_5Aug19_HostCat_RootMstar3_HostRad7_lastroot_tass28_'
          #'noxd')
          # 'OriginMasses_9Aug19_HostCat_RootXTopGal_HostRad22_firstroot_noxd')
           'OriginMasses_20Sep19_HostCat_HostMass14_HostRad16_lastroot_DM')
           #'OriginMasses_9Aug19_HostCat_RootXTopGal_HostRad22_lastroot_noxd')
           #'OriginMasses_8Aug19_HostCat_HostRad7_lastroot_tass28')
           #'OriginMasses_8Aug19_HostCat_HostRad7_RootMass26_firstroot_noxd')
           #'OriginMasses_8Aug19_HostCat_HostRad7_RootMass6_firstroot_XD')
           #'OriginMasses_8Aug19_HostCat_HostRad22_firstroot_noxd')
           #'OriginMasses_5Aug19_HostCat_RootMstar6_HostRad7_firstroot_XDage')

nfiles = 8

stars = False
dm = True

copyfile(dataloc + '.0.hdf5', dataloc + '.hdf5')

if stars:
    mbins_stars = yb.read_hdf5(dataloc + '.hdf5', 'BinnedMasses_Stars')
if dm:
    mbins_dm = yb.read_hdf5(dataloc + '.hdf5', 'BinnedMasses_DM')

if with_parents:
    mbins_stars_par = yb.read_hdf5(dataloc + '.hdf5', 
                                   'BinnedMasses_StarsParent')

for ii in range(1, nfiles):
    filename = dataloc + '.' + str(ii) + '.hdf5'

    if stars:
        mbins_stars_part = yb.read_hdf5(filename, 'BinnedMasses_Stars')
        mbins_stars += mbins_stars_part
    if dm:
        mbins_dm_part = yb.read_hdf5(filename, 'BinnedMasses_DM')
        mbins_dm += mbins_dm_part
        

    if with_parents:
        mbins_starsPar_part = yb.read_hdf5(filename, 
                                           'BinnedMasses_StarsParent')
        mbins_stars_par += mbins_starsPar_part

if combine_sims:
    if stars:
        mbins_stars = np.sum(mbins_stars, axis = 0)
    if dm:
        mbins_dm = np.sum(mbins_dm, axis = 0)
    if with_parents:
        mbins_stars_par = np.sum(mbins_stars_par, axis = 0)

if stars:
    yb.write_hdf5(mbins_stars, dataloc + '.hdf5', 'BinnedMasses_Stars')
if dm:
    yb.write_hdf5(mbins_dm, dataloc + '.hdf5', 'BinnedMasses_DM')
if with_parents:
    yb.write_hdf5(mbins_stars_par, dataloc + '.hdf5', 
                  'BinnedMasses_StarsParent')

print("Done!")
