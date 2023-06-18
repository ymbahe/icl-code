"""
Extract a catalogue with basic information of the clusters for ICL paper.

Started 22 Jul 2019
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
import hydrangea_tools as ht
from pdb import set_trace
import os

outloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/' 
          'cluster_catalogue_z0_22Jul19_merge1p10_wSFR.hdf5')

basedir = '/virgo/simulations/Hydrangea/10r200/' 

isnap = 29    # Snapshot in which clusters are selected

curr_cl = -1

full_sim = np.zeros(1000, dtype = np.int8) - 1
full_gal = np.zeros(1000, dtype = np.int32) - 1

full_m200 = np.zeros(1000) + np.nan
full_r200 = np.zeros(1000) + np.nan
full_mstar_all = np.copy(full_m200)
full_mstar_ca = np.copy(full_m200)
full_mstar_sf = np.copy(full_m200)
full_sfr_30kpc = np.copy(full_m200)

full_relaxTime = np.copy(full_m200)


for isim in range(30):
    
    rundir = basedir + 'CE-{:d}/HYDRO/' .format(isim)
    if not os.path.isdir(rundir): continue

    subdir, snapdir = st.form_files(rundir, isnap, 'sub snap')

    hldir = rundir + 'highlev/'
    fgtloc = hldir + 'FullGalaxyTables.hdf5'
    cantorloc = hldir + 'CantorCatalogue.hdf5'
    posloc = hldir + 'GalaxyPositionsSnap.hdf5'
    spiderloc = hldir + 'SpiderwebTables.hdf5'

    m200_all = yb.read_hdf5(fgtloc, 'M200')
    m200 = m200_all[:, isnap]
    r200 = yb.read_hdf5(fgtloc, 'R200')[:, isnap]

    mergelist = yb.read_hdf5(spiderloc, 'MergeList')

    cenGal_all = yb.read_hdf5(cantorloc, 'CenGalExtended')
    #cenGal_all = yb.read_hdf5(fgtloc, 'CenGal')

    cenGal = cenGal_all[:, isnap]

    contFlag = yb.read_hdf5(fgtloc, 'Full/ContFlag')[:, 1]
    mStar_sf = yb.read_hdf5(fgtloc, 'Mstar')[:, isnap]
    mType_ca = yb.read_hdf5(cantorloc, 'Snapshot_{:03d}/Subhalo/MassType'
                            .format(isnap))
    cantor_shi = yb.read_hdf5(cantorloc, 'SubhaloIndex')[:, isnap]
    sf_shi = yb.read_hdf5(fgtloc, 'SHI')[:, isnap]

    pos_gal_all = yb.read_hdf5(posloc, 'Centre')[:, isnap, :]

    ngal = len(m200)

    # Select galaxies that qualify as 'cluster centrals':
    ind_cl = np.nonzero((m200 > 14.0) & (cenGal == np.arange(ngal)) &
                        (contFlag == 0))[0]

    ncl = len(ind_cl)
    print("There are {:d} clusters in simulation {:d}..." 
          .format(ncl, isim))
    if ncl == 0: continue

    aexp = st.snap_age(snapdir, type = 'aexp')
    conv_astro_pos = aexp/0.6777

    ref_ids_cantor = yb.read_hdf5(cantorloc, 
                                  'Snapshot_{:03d}/IDs' .format(isnap))
    ref_off_cantor = yb.read_hdf5(
        cantorloc, 'Snapshot_{:03d}/Subhalo/OffsetType' .format(isnap))
    ref_end_cantor = yb.read_hdf5(
        cantorloc, 'Snapshot_{:03d}/Subhalo/Extra/OffsetTypeApertures'
        .format(isnap))
    ref_eid_cantor = yb.read_hdf5(
        cantorloc, 'Snapshot_{:03d}/Subhalo/Extra/ExtraIDs' .format(isnap))

    ref_ids_subfind = st.eagleread(subdir, 'IDs/ParticleID', astro = False)
    ref_off_subfind = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
    ref_len_subfind = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)

    snap_ages = ht.snap_times(conv = 'age')
    snap_lbt = snap_ages[isnap] - snap_ages

    for iicl, icl in enumerate(ind_cl):
        
        curr_cl += 1

        r200_cl = r200[icl]
        r200_cl_sim = r200_cl/conv_astro_pos
        
        cl_pos = pos_gal_all[icl, :]
        cl_pos_sim = cl_pos/conv_astro_pos

        readReg = ht.ReadRegion(snapdir, 4, [*cl_pos_sim, r200_cl_sim])
        
        spos = readReg.read_data("Coordinates", astro = True)
        smass = readReg.read_data("Mass", astro = True)
        sid = readReg.read_data("ParticleIDs", astro = False)

        srad_full = np.linalg.norm(spos - cl_pos[None, :], axis = 1)
        ind_in_sphere = np.nonzero((srad_full >= 0.00) 
                                   & (srad_full <= r200_cl))[0]
        
        srad = srad_full[ind_in_sphere]
        smass = smass[ind_in_sphere]
        sid = sid[ind_in_sphere]

        full_sim[curr_cl] = isim
        full_gal[curr_cl] = icl
        full_mstar_all[curr_cl] = np.sum(smass)
        full_m200[curr_cl] = m200[icl]
        full_r200[curr_cl] = r200_cl

        # Match to Cantor particles
        ref_off = ref_off_cantor[cantor_shi[icl], 4]
        ref_end = ref_off_cantor[cantor_shi[icl], 5]
        ref_ids = ref_ids_cantor[ref_off : ref_end]
        
        gate = st.Gate(sid, ref_ids)
        ref_index = gate.in2()
        ind_in_ca = np.nonzero(ref_index >= 0)[0]

        full_mstar_ca[curr_cl] = np.sum(smass[ind_in_ca])
        
        # Match to Subfind particles
        ref_off = ref_off_subfind[sf_shi[icl]]
        ref_end = ref_off + ref_len_subfind[sf_shi[icl]]
        ref_ids = ref_ids_subfind[ref_off : ref_end]
        
        gate = st.Gate(sid, ref_ids)
        ref_index = gate.in2()
        ind_in_sf = np.nonzero(ref_index >= 0)[0]

        full_mstar_sf[curr_cl] = np.sum(smass[ind_in_sf])

        # Same for SFR (CA only)
        readReg = ht.ReadRegion(snapdir, 0, [*cl_pos_sim, 0.03/conv_astro_pos])
        
        gpos = readReg.read_data("Coordinates", astro = True)
        gsfr = readReg.read_data("StarFormationRate", astro = True)
        gid = readReg.read_data("ParticleIDs", astro = False)
         
        # Match to Cantor particles
        if ref_eid_cantor[cantor_shi[icl]] < 0: set_trace()
        ref_off = ref_off_cantor[cantor_shi[icl], 0]
        ref_end = ref_end_cantor[ref_eid_cantor[cantor_shi[icl]], 0, 2]
        ref_ids = ref_ids_cantor[ref_off : ref_end]

        gate = st.Gate(gid, ref_ids)
        ref_index = gate.in2()
        ind_in_ca = np.nonzero(ref_index >= 0)[0]

        full_sfr_30kpc[curr_cl] = np.sum(gsfr[ind_in_ca])
        
       
        # Find time of last major halo merger
        for isnap_back in range(29, 0, -1):
            
            ind_acc = np.nonzero(
                (cenGal_all[mergelist[:, isnap_back], isnap_back] == icl) &
                (cenGal_all[:, isnap_back-1] == np.arange(ngal)) &
                (np.arange(ngal) != icl))[0]
        
            if len(ind_acc) == 0: continue

            m200_pre = np.max(m200_all[ind_acc, isnap_back-1])
            m200_self = m200_all[icl, isnap_back-1]

            print("Max merger in snap {:d}: {:.2f} vs. {:.2f}" 
                  .format(isnap_back, m200_pre, m200_self))

            if m200_pre >= m200_self + np.log10(1/10):
                print("Major merger in snap {:d} (M200 = {:.2f} vs. {:.2f})"
                      .format(isnap_back, m200_pre, m200_self))
                full_relaxTime[curr_cl] = snap_lbt[isnap_back] + (
                    snap_lbt[isnap_back-1]-snap_lbt[isnap_back])/2
                break

        # Ends loop through snapshots for finding last major merger
    # Ends loop through clusters
# Ends loop through simulations

yb.write_hdf5(full_sim[:curr_cl], outloc, 'Sim', new = True)
yb.write_hdf5(full_gal[:curr_cl], outloc, 'Galaxy')
yb.write_hdf5(full_m200[:curr_cl], outloc, 'M200c')
yb.write_hdf5(full_mstar_all[:curr_cl], outloc, 'MstarAll')
yb.write_hdf5(full_mstar_ca[:curr_cl], outloc, 'MstarCantor')
yb.write_hdf5(full_mstar_sf[:curr_cl], outloc, 'MstarSubfind')
yb.write_hdf5(full_relaxTime[:curr_cl], outloc, 'RelaxTime')
yb.write_hdf5(full_sfr_30kpc[:curr_cl], outloc, 'SFR_30kpc')
yb.write_hdf5(full_r200[:curr_cl], outloc, 'R200c')


print("Done!")
