"""
Extract example particles for Cantor plots
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import hydrangea as hy

final_only = False

rundir = '/virgo/simulations/Hydrangea/10r200/CE-28/HYDRO/'
snapdir, subdir = st.form_files(rundir, 29, 'snap sub')

gpsloc = rundir + 'highlev/GalaxyPositionsSnap.hdf5'
fgtloc = rundir + 'highlev/FullGalaxyTables.hdf5'

outloc = '/virgo/scratch/ybahe/HYDRANGEA/TESTS/cantor_demo_spos_all.hdf5'

galpos = yb.read_hdf5(gpsloc, 'Centre')[3272, 29, :]
clpos = yb.read_hdf5(gpsloc, 'Centre')[474, 29, :] - galpos
shi = yb.read_hdf5(fgtloc, 'SHI')[3272, 29]

readReg = hy.ReadRegion(snapdir, 4, [*galpos, 1.0], verbose=True)
spos = readReg.read_data('Coordinates', verbose=True) - galpos[None, :]
sids = readReg.read_data('ParticleIDs')

cshi = yb.read_hdf5(rundir + 'highlev/Cantor/GalaxyTables.hdf5',
                    'SubhaloIndex')
csi = cshi[3272, 29]
csp = cshi[3272, 28]
csr = cshi[3272, 6]

if final_only:
    print("Finding self-bound particles...")
    cantor_ids = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_029_IDs.hdf5',
                               'IDs')
    cantor_off = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_029.hdf5',
                               'Subhalo/Offset')
    cantor_len = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_029.hdf5',
                               'Subhalo/Length')
    cantor_ids_gal = cantor_ids[cantor_off[csi]:
                            cantor_off[csi]+cantor_len[csi]]
    inds, ind_gal = hy.crossref.Gate(sids, cantor_ids_gal).in_int()
    spos = spos[ind_gal, :]
    sids = sids[ind_gal]

ids_all = st.eagleread(subdir, 'IDs/ParticleID', astro=False)
offsets = st.eagleread(subdir, 'Subhalo/SubOffset', astro=False)
lengths = st.eagleread(subdir, 'Subhalo/SubLength', astro=False)
ids_gal = ids_all[offsets[shi]:offsets[shi]+lengths[shi]]

inds, ind_sf = hy.crossref.Gate(sids, ids_gal).in_int()
ids_sf = sids[ind_sf]

cantor_ids_prev = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_028_IDs.hdf5',
                               'IDs')
cantor_off_prev = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_028.hdf5',
                               'Subhalo/Offset')
cantor_len_prev = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_028.hdf5',
                               'Subhalo/Length')
cantor_ids_gal_prev = cantor_ids_prev[cantor_off_prev[csp]:
                                      cantor_off_prev[csp]+cantor_len_prev[csp]]



inds, ind_cp = hy.crossref.Gate(sids, cantor_ids_gal_prev).in_int()
ids_cp = sids[ind_cp]

cantor_ids_ref = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_006_IDs.hdf5',
                              'IDs')
cantor_off_ref = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_006.hdf5',
                              'FOF/Offset')
cantor_len_ref = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_006.hdf5',
                              'FOF/Length')
ifof = yb.read_hdf5(rundir + 'highlev/Cantor/Cantor_006.hdf5',
                    'Subhalo/FOF_Index')[csr]

cantor_ids_gal_ref = cantor_ids_ref[cantor_off_ref[ifof]:
                                    cantor_off_ref[ifof]+cantor_len_ref[ifof]]
inds, ind_ref = hy.crossref.Gate(sids, cantor_ids_gal_ref).in_int()
ids_ref = sids[ind_ref]

mask = np.zeros(len(sids), dtype = np.int8)
mask[ind_sf] = 1
mask[ind_cp] = 1
mask[ind_ref] = 1
ind_sel = np.nonzero(mask == 1)[0]
sids_sel = sids[ind_sel]

subind_sf, ind_found_sf = hy.crossref.Gate(ids_sf, sids_sel).in_int()
subind_cp, ind_found_cp = hy.crossref.Gate(ids_cp, sids_sel).in_int()
subind_ref, ind_found_ref = hy.crossref.Gate(ids_ref, sids_sel).in_int()

if len(ind_found_sf) != len(subind_sf): set_trace()
if len(ind_found_cp) != len(subind_cp): set_trace()
if len(ind_found_ref) != len(subind_ref): set_trace()

yb.write_hdf5(spos[ind_sel, :], outloc, 'Coordinates', new=True)
yb.write_hdf5(subind_sf, outloc, 'IndSF')
yb.write_hdf5(subind_cp, outloc, 'IndCP')
yb.write_hdf5(subind_ref, outloc, 'IndREF')
yb.write_hdf5(clpos, outloc, 'ClPos')

print("Done!")
