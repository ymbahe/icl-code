"""
Collect the age/metallicity/density profiles generated for individual clusters.
"""

import numpy as np
import yb_utils as yb
import os
from astropy.io import ascii
from pdb import set_trace

#sims = np.array([12, 18, 22, 22, 22, 24, 25, 25, 28, 29]) 
#gals = np.array([  280,   830,     4,   652, 26044,     0,   248,  1148,   474,
#                   10859])
#m200 = np.array([14.600133 , 14.733843 , 14.655903 , 14.904098 , 14.611923 ,
#                 14.892035 , 14.692819 , 14.721917 , 14.8713665, 14.960957 ])

scriptloc = '/u/ybahe/ANALYSIS/IMAGES/cluster_script_s29_xt.sh'

scriptdata = ascii.read(scriptloc)
sims = np.array(scriptdata['col4'])
gals = np.array(scriptdata['col5'])
m200 = np.array(scriptdata['col6'])
r200 = np.array(scriptdata['col7'])

outloc = '/virgo/scratch/ybahe/RESULTS/ICL/Profiles_Cantor_z0p0_des_gri.hdf5'

n_cl = len(sims)
n_bin = 31

m_all = np.zeros((n_cl, n_bin))
z_all = np.zeros((n_cl, n_bin))
a_all = np.zeros((n_cl, n_bin))
rr_all = np.zeros((n_cl, n_bin))
rp_all = np.zeros((n_cl, n_bin+1))

lum_all_g = np.zeros((n_cl, n_bin))
lum_all_r = np.zeros((n_cl, n_bin))
lum_all_i = np.zeros((n_cl, n_bin))

flag_all = np.zeros(n_cl, dtype = int)

imtype = 'gri'

for icl in range(n_cl):

    isim = sims[icl]
    igal = gals[icl]
    
    sizecode = '{:.2f}' .format(r200[icl])
    sizecode = sizecode.replace('.', 'p')
    sizecode = '3p00'

    dataloc = ('/virgo/scratch/ybahe/RESULTS/ICL/PROFILE_DATA/'
               'Cluster_CE-{:d}_G-{:d}x_PT-4_{:s}_IR_gri_'
               'ProfilesOnly0029.hdf5' .format(isim, igal, sizecode))

    print(dataloc)
    if not os.path.exists(dataloc): continue

    print("Found file for cluster {:d}." .format(icl))

    set_trace()

    flag_all[icl] = 1

    rlim_mpc = yb.read_hdf5(dataloc, 'Edge_Annuli')
    rlim = rlim_mpc/r200[icl]

    r_bin = yb.read_hdf5(dataloc, 'Rad_Annuli')/r200[icl]

    if imtype == 'gri':
        lum_all_g[icl, :] = yb.read_hdf5(dataloc, 'Luminosity_g')
        lum_all_r[icl, :] = yb.read_hdf5(dataloc, 'Luminosity_r')
        lum_all_i[icl, :] = yb.read_hdf5(dataloc, 'Luminosity_i')
    else:
        m_all[icl, :] = yb.read_hdf5(dataloc, 'Mass_Annuli')
        z_all[icl, :] = yb.read_hdf5(dataloc, 'FeH_Annuli')
        a_all[icl, :] = yb.read_hdf5(dataloc, 'Age_Annuli')

    rr_all[icl, :] = r_bin
    rp_all[icl, :] = rlim_mpc

yb.write_hdf5(rlim, outloc, 'Edge_Annuli')
yb.write_hdf5(rp_all, outloc, 'Edge_Annuli_Phys')
yb.write_hdf5(rr_all, outloc, 'Rad_Annuli_Rel')

if imtype == 'gri':
    yb.write_hdf5(lum_all_g, outloc, 'Lum_g_Annuli')
    yb.write_hdf5(lum_all_r, outloc, 'Lum_r_Annuli')
    yb.write_hdf5(lum_all_i, outloc, 'Lum_i_Annuli')
else:
    yb.write_hdf5(m_all, outloc, 'Mass_Annuli')
    yb.write_hdf5(z_all, outloc, 'FeH_Annuli')
    yb.write_hdf5(a_all, outloc, 'Age_Annuli')


yb.write_hdf5(flag_all, outloc, 'Flag')
yb.write_hdf5(m200, outloc, 'M200')

print("Done!")
