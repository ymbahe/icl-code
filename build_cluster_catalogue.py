"""
Extract a catalogue with basic information of the clusters for ICL paper.

Started 22 Jul 2019
"""

import numpy as np
import hydrangea as hy
import hydrangea.hdf5 as hd
from pdb import set_trace
import os

outloc = ('cluster_catalogue_z0_20Jun23_merge1p10_wSFR.hdf5')

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
    if isim in [10, 17, 19, 20, 23, 26, 27]:
        continue
    print(f"Processing simulation {isim}...")
    
    sim = hy.Simulation(isim)
    subdir = sim.get_subfind_file(isnap)
    snapdir = sim.get_snapshot_file(isnap)

    hldir = sim.high_level_dir
    cantor_tableloc = hldir + 'Cantor/GalaxyTables.hdf5'
    cantor_file = hldir + f'Cantor/Cantor_{isnap:03d}.hdf5'
    cantor_id_file = hldir + f'Cantor/Cantor_{isnap:03d}_IDs.hdf5'

    print("   ... reading data...")
    m200_all = hy.hdf5.read_data(sim.fgt_loc, 'M200')
    m200 = m200_all[:, isnap]
    r200 = hy.hdf5.read_data(sim.fgt_loc, 'R200')[:, isnap]

    mergelist = hy.hdf5.read_data(sim.spider_loc, 'MergeList')
    cenGal_all = hy.hdf5.read_data(cantor_tableloc, 'CentralGalaxy')
    cenGal = cenGal_all[:, isnap]

    contFlag = hy.hdf5.read_data(sim.fgt_loc, 'Full/ContFlag')[:, 1]
    mStar_sf = hy.hdf5.read_data(sim.fgt_loc, 'Mstar')[:, isnap]
    mType_ca = hy.hdf5.read_data(cantor_file, 'Subhalo/MassType') * 1e10
    pos_ca = hy.hdf5.read_data(cantor_file, 'Subhalo/CentreOfPotential')
    
    cantor_shi = hy.hdf5.read_data(cantor_tableloc, 'SubhaloIndex')[:, isnap]
    sf_shi = hy.hdf5.read_data(sim.fgt_loc, 'SHI')[:, isnap]

    pos_gal_all = hy.hdf5.read_data(sim.gps_loc, 'Centre')[:, isnap, :]

    ngal = len(m200)

    # Select galaxies that qualify as 'cluster centrals':
    ind_cl = np.nonzero(
        (m200 > 14.0) & (cenGal == np.arange(ngal)) & (contFlag == 0))[0]
    ncl = len(ind_cl)
    print(f"There are {ncl} clusters in simulation {isim}...")
    if ncl == 0: continue

    ref_ids_cantor = hd.read_data(cantor_id_file, "IDs")
    ref_off_cantor = hd.read_data(cantor_file, 'Subhalo/OffsetType')
    ref_end_cantor = hd.read_data(
        cantor_file, 'Subhalo/Extra/OffsetTypeApertures')
    ref_eid_cantor = hd.read_data(cantor_file, 'Subhalo/Extra/ExtraIDs')

    ref_ids_subfind = hy.SplitFile(subdir, 'IDs').ParticleID
    sub = hy.SplitFile(subdir, 'Subhalo')
    ref_off_subfind = sub.SubOffset
    ref_len_subfind = sub.SubLength

    snap_lbt = hy.snep_times(time_type='lbt', snep_list='allsnaps')

    for iicl, icl in enumerate(ind_cl):

        print(f"   ... processing galaxy {icl}...")
        curr_cl += 1
        r200_cl = r200[icl]
        cshi = cantor_shi[cenGal[icl]]
        cl_pos = pos_ca[cshi, :]
        stars = hy.ReadRegion(snapdir, 4, cl_pos, r200_cl, exact=True)
        
        full_sim[curr_cl] = isim
        full_gal[curr_cl] = icl
        full_mstar_all[curr_cl] = np.sum(stars.Mass)
        full_m200[curr_cl] = m200[icl]
        full_r200[curr_cl] = r200_cl

        sid = stars.ParticleIDs
        
        # Match to Cantor particles
        ref_off = ref_off_cantor[cshi, 4]
        ref_end = ref_off_cantor[cshi, 5]
        ref_ids = ref_ids_cantor[ref_off : ref_end]
        
        ref_index, ind_matched = hy.crossref.find_id_indices(
            stars.ParticleIDs, ref_ids)
        full_mstar_ca[curr_cl] = np.sum(stars.Mass[ind_matched])
        
        # Match to Subfind particles
        ref_off = ref_off_subfind[sf_shi[icl]]
        ref_end = ref_off + ref_len_subfind[sf_shi[icl]]
        ref_ids = ref_ids_subfind[ref_off : ref_end]
        
        ref_index, ind_matched = hy.crossref.find_id_indices(
            stars.ParticleIDs, ref_ids)
        full_mstar_sf[curr_cl] = np.sum(stars.Mass[ind_matched])

        # Same for SFR (CA only)
        gas = hy.ReadRegion(snapdir, 0, cl_pos, 0.03, exact=True)
        gsfr = gas.StarFormationRate
                 
        # Match to Cantor particles
        if ref_eid_cantor[cshi] < 0: set_trace()
        ref_off = ref_off_cantor[cshi, 0]
        ref_end = ref_end_cantor[ref_eid_cantor[cshi], 0, 2]
        ref_ids = ref_ids_cantor[ref_off : ref_end]

        ref_index, ind_matched = hy.crossref.find_id_indices(
            gas.ParticleIDs, ref_ids)
        full_sfr_30kpc[curr_cl] = np.sum(gas.StarFormationRate[ind_matched])
               
        # Find time of last major halo merger
        for isnap_back in range(29, 0, -1):
        
            # Find galaxies accreted to FOF in last interval
            ind_acc = np.nonzero(
                (cenGal_all[mergelist[:, isnap_back], isnap_back] == icl) &
                (cenGal_all[:, isnap_back-1] == np.arange(ngal)) &
                (np.arange(ngal) != icl)
            )[0]
            if len(ind_acc) == 0: continue

            m200_pre = np.max(m200_all[ind_acc, :isnap_back])
            m200_self = m200_all[icl, isnap_back-1]
            print(f"Max merger in snap {isnap_back}: "
                  f"{m200_pre:.2f} vs. {m200_self:.2f}") 

            if m200_pre >= m200_self + np.log10(1/10):
                print(f"Major merger in snap {isnap_back}!")
                full_relaxTime[curr_cl] = snap_lbt[isnap_back] + (
                    snap_lbt[isnap_back-1] - snap_lbt[isnap_back]) / 2

                # We can stop here, because we go backwards through snaps
                break

        # Ends loop through snapshots for finding last major merger
    # Ends loop through clusters
# Ends loop through simulations

hd.write_data(outloc, 'Sim', full_sim[:curr_cl], new=True)
hd.write_data(outloc, 'Galaxy', full_gal[:curr_cl])
hd.write_data(outloc, 'M200c', full_m200[:curr_cl])
hd.write_data(outloc, 'MstarAll', full_mstar_all[:curr_cl])
hd.write_data(outloc, 'MstarCantor', full_mstar_ca[:curr_cl])
hd.write_data(outloc, 'MstarSubfind', full_mstar_sf[:curr_cl])
hd.write_data(outloc, 'RelaxTime', full_relaxTime[:curr_cl])
hd.write_data(outloc, 'SFR_30kpc', full_sfr_30kpc[:curr_cl])
hd.write_data(outloc, 'R200c', full_r200[:curr_cl])

print("Done!")
