"""
Extract "birth galaxy" of all stellar (and optionally other) particles.

'Extra-detailed' version that also splits by birth time and birth radius.

Starting with the first snapshot in which the particle exists (in its
type, for stars/BHs), the script tests whether it belongs to a galaxy, which is
then its 'root'. Particles that are not associated to a galaxy are re-tested
in the subsequent snapshot, until z = 0 if necessary.

Optionally, a galaxy hosting the particle can be discarded if its mass is
below an adjustable threshold in units of the particle's z = 0 galaxy mass,
either at z = 0 or in the snapshot under consideration.

For stars/BHs, also the root galaxy of the parent gas particle is determined,
in an analogous way but starting from snapshot 0.

This version is modified to use Cantor catalogues as input.

Output:

PartType[x]/ParticleIDs  --> Particle IDs, for matching to other data sets
PartType[x]/RootSnapshot --> First snapshot in which particle is in subhalo
PartType[x]/RootGalaxy   --> Galaxy of particle in root snapshot

For stars and BHs only:
PartType[x]/ParentRootSnapshot --> As RootSnapshot, but for gas parent particle
PartType[x]/ParentRootGalaxy   --> As RootGalaxy, but for gas parent particle

 -- Started 2-Feb-2018
 -- Updated 9-May-2019: improving documentation and removing any involvement
                        of eagle_subfind_particles tables. Also large-scale
                        re-structuring.
 -- Updated 25-Jun-19:  modifications to use Cantor catalogues as input

 -- Updated 30-Jul-19:  use catalogue of input galaxies (=clusters), and 
                        only output properties for these.

"""

import numpy as np
import sim_tools as st
import yb_utils as yb
from astropy.io import ascii
from pdb import set_trace
import calendar
import time
from mpi4py import MPI
import os
import hydrangea_tools as ht
import eagle_routines as er
import sys
import ctypes as c
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13, z_at_value

simname = "Hydrangea"     # 'Eagle' or 'Hydrangea'
runtype = "HYDRO"         # 'HYDRO' or 'DM'
ptypeList = [1]           # List of particle types to process

# File name of the Cantor Catalogue to use:
cantorCatalogue = 'CantorCatalogue.hdf5' 

# File name of *cluster* catalogue to use:
clusterCatalogue = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/' 
                    'cluster_catalogue_z0_22Jul19.hdf5')

# Next line specifies the minimum mass of a subhalo to be considered as 
# 'containing' a particle, in units of the particle's z = 0 host.
# If it is None, no limit is applied.
min_ratio = None

# Specify the point at which subhalo masses are compared (only relevant if 
# min_ratio != None). 'current' (at snapshot itself) or 'z0' (at z = 0).
comp_type = 'current'

# Also determine the analogous root galaxy/snapshot of particles' parent
# gas particles (only for stars/BHs)?
include_parents = False

# Delete ('forget') parent gas root galaxy if the particle is unbound
# in any snapshot prior to turning into a star/BH? Note, this only makes
# sense in combination with rootType 'last'.
forgetful_parents = True

# This currently does nothing. In principle, it would be useful to make this
# work as it may provide a better 'progenitor mass'.
rootType = 'last'  # 'first' (first SH association) or 'last' (before final) 

# Record assembly time instead of formation time?
record_assembly = True

# --------- Options for origin code determination -------------

# Threshold between merged and quasi-stripped, in log (M_star/M_sun)
# (set to None to not apply any threshold, then no quasi-stripped):
max_mstar_qm = 9.0   


# --------- Options for output catalogue ----------------------

# Minimum and maximum mass of root galaxy bins:
root_mass_range = [9.0, 15.0]
root_mass_type = 'msub'
root_mass_nbins = 14   # incl. 2 for outliers

# Min and max radius in host (in log Mpc), and number of bins:
host_lograd_range = [-3, 0.5]
host_lograd_nbins = 16   # incl. 2 for outliers

# Min and max rel-rad, and number of bins:
host_relrad_range = [0, 10]
host_relrad_nbins = 2     # incl. 1 for outliers (no negative relRad!)

# Min and max age (in Gyr), and number of bins:
age_range = [0, 14]
age_nbins = 1     # no outliers here

# Min and max birth radius (in log Mpc or SHMR), and number of bins:
root_lograd_range = [0, 5]
root_lograd_nbins = 3     # incl. 2 outliers (for consistency)

# Scale root radius by stellar half mass radius?
scale_root_lograd = False

# Use initial mass for stars (and initial baryon mass for gas)? If False,
# the mass at z = 0 will be used.
use_initial_mass = False  
 
# Count only metal mass (False: total stellar mass)?
use_metal_mass = False

# Limit parent tracing to x Gyr prior to star formation time?
parent_limit = 2.0

# =======================================================================

if simname == "Eagle":
    n_sim = 1
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    snap_z0 = 28
    nsnap = 29
    snapAexpLoc = ('/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/' + 
                   'eagle_outputs_new.txt')
else:    
    n_sim = 30
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    snap_z0 = 29
    nsnap = 30
    snapAexpLoc = ('/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/' +
                   'hydrangea_snapshots_plus.dat')


outloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/' +
          'OriginMasses_20Sep19_HostCat_HostMass14_HostRad16_lastroot_DM')
          #'OriginMasses_19Sep19_HostCat_HostRad22_lastroot_withForgetfulParents_lim2p0Gyr')

#partloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/OriginParticles_19Sep19_CE28_TEST.hdf5')

# =============================================================================


def find_root_snap(iiptype):
    """Find the root snapshot for particles of a given type"""

    print("Find birth snapshot...", flush = True)
    
    if iiptype not in [4, 5]: return

    if iiptype == 4:
        formAexp = st.eagleread(snapdir_z0, 'PartType4/StellarFormationTime', 
                                astro = False, silent = True)
    else:
        formAexp = st.eagleread(snapdir_z0, 'PartType5/BH_FormationTime', 
                                astro = False, silent = True)
        
    # Find first snapshot AFTER birth. In the event that there is one
    # exactly at the birth time, pick that one (hence side = 'left')
    
    return np.searchsorted(snap_aexp, formAexp, side = 'left').astype(np.int8)


def find_galaxy_for_ids(rundir, ids, isnap, ptype_offset=None, ptype=None,
                        check_cont=False, return_rad = False):
    """
    Identify the galaxy hosting input particle IDs in a given snapshot.

    ptype_offset specifies the first particle of type x. For particles
    beyond ptype_offset 6, all particles are considered.
    """

    pre = 'Snapshot_' + str(isnap).zfill(3)
    ca_ids = yb.read_hdf5(cantorloc, pre + '/IDs')

    ca_offset = yb.read_hdf5(cantorloc, pre + '/Subhalo/Offset')
    ca_length = yb.read_hdf5(cantorloc, pre + '/Subhalo/Length')
    
    ca_offset_type = yb.read_hdf5(
        cantorloc, pre + '/Subhalo/OffsetType')
    ca_length_type = ca_offset_type[:, 1:]-ca_offset_type[:, :-1]

    # The next lines are robust to the presence of particles 
    # outside the Cantor ID list or subhaloes
    ca_inds, ind_found_ca = yb.find_id_indices(ids, ca_ids)

    ca_shi = np.zeros(len(ids), dtype = np.int32)-1
    
    # First mode: all particles are of one specified (real) type:
    if ptype is not None:
        ca_shi = st.ind_to_sh(ca_inds, ca_offset_type[:, ptype],
                              ca_length_type[:, ptype])

    # Second mode: combined particle list, treat different types separately
    elif ptype_offset is not None:

        # Real types: consider only entries for this type
        for iptype in range(6):
            if ptype_offset[iptype+1] > ptype_offset[iptype]:
                ca_shi[ptype_offset[iptype]:ptype_offset[iptype+1]] = (
                    st.ind_to_sh(ca_inds[ptype_offset[iptype]:
                                         ptype_offset[iptype+1]],
                                 ca_offset_type[:, iptype], 
                                 ca_length_type[:, iptype]))
    
        # Parent types: consider all entries within galaxy
        ca_shi[ptype_offset[6]:] = (
            st.ind_to_sh(ca_inds[ptype_offset[6]:],
                         ca_offset, ca_length))

    # Third mode: no particle types, consider all entries for all
    else:
        ca_shi = st.ind_to_sh(ca_inds, ca_offset, ca_length)

    # Last bit is mode-independent: find particles in SH, and 
    # convert SHI --> galaxy for these.

    ca_shi_gal = yb.read_hdf5(cantorloc, pre + '/Subhalo/Galaxy')
    ind_in_gal = np.nonzero(ca_shi >= 0)[0]

    ca_gal = np.zeros_like(ca_shi)-1
    ca_gal[ind_in_gal] = ca_shi_gal[ca_shi[ind_in_gal]]

    if check_cont:
        subind_clean = np.nonzero(contFlag[ca_gal[ind_in_gal]] == 0)[0]
        ind_in_gal = ind_in_gal[subind_clean]

    if return_rad:
        rad = yb.read_hdf5(cantorloc, pre + '/Radius')
        rad_gal = np.zeros(ca_gal.shape[0], dtype = np.float32)+1000
        rad_gal[ind_in_gal] = rad[ca_inds[ind_in_gal]]
        return ca_gal, ind_in_gal, rad_gal
    else:
        return ca_gal, ind_in_gal
    

def load_particles():
    """
    Load all particles of desired type(s) in galaxies at z = 0.

    For stars/BHs, parent gas particles are included as duplicate,
    with ptype = 6/7 (i.e. real ptype + 2).
    
    Returns:
    --------
    all_ids : ndarray (int)
        The IDs of particles to process
    all_gal : ndarray (int)
        The galaxy to which each particle belongs at z = 0.
    ptype_offset : ndarray (int) [9]
        The offsets of individual particle types in the ID list.
    """ 

    all_ids = np.zeros(0, dtype = int)
    all_gal = np.zeros(0, dtype = int)
    all_ft = np.zeros(0, dtype = float)
    ptype_offset = np.zeros(9, dtype = int)

    # Loop through individual (to-be-considered) particle types:
    for iiptype in ptypeList:
    
        # Load all IDs at z = 0:
        ptype_ids = st.eagleread(snapdir_z0, 'PartType{:d}/ParticleIDs' 
                                 .format(iiptype), 
                                 astro = False, silent = True).astype(int)

        if iiptype == 4:
            ptype_ft = st.eagleread(snapdir_z0, 
                                    'PartType{:d}/StellarFormationTime'
                                    .format(iiptype),
                                    astro = False, silent = True)
        else:
            ptype_ft = np.zeros(ptype_ids.shape[0])

        # Identify their z = 0 host galaxy:
        ptype_gal, ptype_in_gal = find_galaxy_for_ids(
            rundir, ptype_ids, snap_z0, ptype = iiptype, check_cont=True)

        # Extra step: check which galaxies are useful (in catalogue),
        # and discard others
        subind_in_cat = np.nonzero(catInd[ptype_gal[ptype_in_gal]] >= 0)[0]
        ptype_in_cat = ptype_in_gal[subind_in_cat]

        # Now append matches to full list:
        all_ids = np.concatenate((all_ids, ptype_ids[ptype_in_cat]))
        all_gal = np.concatenate((all_gal, ptype_gal[ptype_in_cat]))
        all_ft = np.concatenate((all_ft, ptype_ft[ptype_in_cat]))
        ptype_offset[iiptype+1:] += len(ptype_in_cat)

    # If parent-finding is included, we simply duplicate the required IDs.
    # That way, we don't have to include parents as separate category in the 
    # main parts of the program.
    if include_parents:
        for iiParType in [4, 5]:
            if iiParType in ptypeList: 
                all_ids = np.concatenate(
                    (all_ids, 
                     all_ids[ptype_offset[iiParType]:ptype_offset[iiParType+1]]))
                all_gal = np.concatenate(
                    (all_gal,
                     all_gal[ptype_offset[iiParType]:ptype_offset[iiParType+1]]))
                all_ft = np.concatenate(
                    (all_ft,
                     all_ft[ptype_offset[iiParType]:ptype_offset[iiParType+1]]))

                # Increase offsets for last `types' (stored beyond pt-5):    
                ptype_offset[iiParType+3:] += (
                    ptype_offset[iiParType+1]-ptype_offset[iiParType])
                
    return all_ids, all_gal, all_ft, ptype_offset


def process_snapshot(rundir, isnap, cantorloc):
    """Process one snapshot for root-galaxy memberships..."""

    print("")
    print("---------- Starting snapshot = {:d} (sim = {:d}) ----------- " 
          .format(isnap, isim))
    print("")
    sys.stdout.flush()

    # Need to consider all particles that have not yet been marked as 
    # 'found' (set rootSnap --> 1)
    
    if rootType == 'first':
        ind_part_ts = np.nonzero(all_rootSnap == isnap)[0]
    else:
        if parent_limit is None:
            ind_part_ts = np.arange(len(all_rootSnap))
        else:
            snapdir = st.form_files(rundir, isnap, 'snap')
            curr_time = st.snap_age(snapdir, 'age')
            if curr_time+parent_limit < age_now-0.05:
                limit_time = curr_time+parent_limit
                curr_formation_zlimit = z_at_value(Planck13.age, limit_time)
                curr_formation_alimit = 1/(1+curr_formation_zlimit)
            else:
                curr_formation_alimit = 1.0
            
            ind_part_ts = np.nonzero(all_ft < curr_formation_alimit)[0]

    np_snap = len(ind_part_ts)

    if np_snap == 0:
        return

    print("Processing {:d} particles in snapshot {:d}..." 
          .format(np_snap, isnap))

    # Extract current gal-->cantorID list, for simplicity:
    cantorIDs = cantor_shi[:, isnap]

    # Set up a mask to record which particles could be matched
    # (0 --> no, 1 --> yes)
    ts_flag_attached = np.zeros(np_snap, dtype = np.int8)

    # Key part:
    # Identify the galaxy (if any) of all currently-considered particles
    # Prefix 'sn' for 'this snap':
    ts_gal, ts_ind_in_gal, rad_gal = find_galaxy_for_ids(
        rundir, all_ids[ind_part_ts], isnap, ptype_offset, return_rad = True)

    np_gal = len(ts_ind_in_gal)
    print("... of which {:d} are in a galaxy..." .format(np_gal))
    if np_gal == 0: return

    # Now check which identifications are 'valid', which depends on the 
    # program settings (prefix 'ig' --> 'in_gal')

    # Load total subhalo mass, which is used as proxy
    if root_mass_type == 'msub':
        msub = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(isnap).zfill(3) + '/Subhalo/Mass')
    elif root_mass_type == 'mstar':
        msub = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(isnap).zfill(3) + 
            '/Subhalo/MassType')[:, 4]
    else:
        print("Invalid root mass type '" + root_mass_type + "'")
        set_trace()

    msub = np.log10(msub)+10.0

    if rootType == 'first':
        # Simple case: attach particle only to first available galaxy
        # (in this case we only test not-yet-attached particles anyway):
        subind_permitted = np.arange(np_gal)
 
    else:
        # In this case, we need to check whether particles can be attached
        # to their current host:
        # a) Not yet attached --> always attach to current host
        # b) Attached to z=0 host --> only attach if to z=0 host
        # c) Attached to other gal --> only attach is NOT to z=0 host
        
        # Use prefix 'tsg' --> this snap, in gal
        tsg_gal = ts_gal[ts_ind_in_gal]
        tsg_old_root = all_rootGal[ind_part_ts[ts_ind_in_gal]]
        tsg_z0gal = all_gal_z0[ind_part_ts[ts_ind_in_gal]]
        tsg_msub = msub[cantorIDs[tsg_gal]]
        tsg_root_mass = all_rootMass[ind_part_ts[ts_ind_in_gal]]
        if record_assembly:
            tsg_tass = all_tass[ind_part_ts[ts_ind_in_gal]]

        # Conditions below implicitly exclude galaxies not in z=0 gal,
        # 
        
        subind_permitted = np.nonzero(
            (tsg_old_root < 0) | ((tsg_msub > tsg_root_mass) & 
            ((tsg_old_root == tsg_z0gal) & (tsg_gal == tsg_z0gal)) |
            ((tsg_old_root != tsg_z0gal) & (tsg_gal != tsg_z0gal))
                              ))[0]

        # New test: first assigned to root? Record time:
        if record_assembly:
            snapdir = st.form_files(rundir, isnap, 'snap')
            curr_time = st.snap_age(snapdir, 'age')
            ind_assembled = np.nonzero(
                (tsg_tass < 0) & (tsg_gal == tsg_z0gal))[0]
            tsg_tass[ind_assembled] = curr_time
            all_tass[ind_part_ts[ts_ind_in_gal]] = tsg_tass

    # Second criterion: host massive enough?
    if min_ratio is None:

        # Simple case: any host will do.
        subind_massive = np.arange(len(subind_permitted))

    else:
        # Not-so-simple case: need to check which hosts are massive enough.

        # Two ways of testing 'massive enough': 
        # (i) Based on current masses
        if comp_type == 'current':

            # Need to first find comparison (`ask') mass of z = 0 host:
            all_compMass = np.zeros(numPart)

            # Determine z0-host's SHI in current snap (all)
            all_z0Gal_currSHI = cantorIDs[all_gal_z0]
            
            # Check whose z0-gals are alive in current snap
            # (NB: ind_prog_alive also excludes not-in-z0-host particles):
            ind_prog_alive = np.nonzero(all_z0Gal_currSHI >= 0)[0]

            curr_mass_alive = msub[all_z0Gal_currSHI[ind_prog_alive]]
            all_compMass[ind_prog_alive] = curr_mass_alive

            # Now check which (permitted) particles are massive hosts:
            subind_massive = np.nonzero(
                msub[cantorIDs[ts_gal[ts_ind_in_gal[subind_permitted]]]]
                >= min_ratio * all_compMass[ind_part_ts[ts_ind_in_gal[
                    subind_permitted]]])[0]

        # (ii) Based on z = 0 masses (this is simpler)
        else:
            subind_massive = np.nonzero(
                msub_z0[cantor_shi[ts_gal[ts_ind_in_gal[subind_permitted]], 
                                   snap_z0]] 
                >= min_ratio * all_compMass[ind_part_ts[ts_ind_in_gal[
                    subind_permitted]]])[0]

    # Almost done! 
    ts_ind_attached = ts_ind_in_gal[subind_permitted[subind_massive]]
    ts_flag_attached[ts_ind_attached] = 1
    ts_ind_unmatched = np.nonzero(ts_flag_attached == 0)[0]

    print("Could match {:d} particles to galaxies ({:d} unmatched)."
          .format(len(subind_massive), len(ts_ind_unmatched)), flush = True)

    # Update root galaxy and mass for (re-)attached galaxies:
    all_rootGal[ind_part_ts[ts_ind_attached]] = ts_gal[ts_ind_attached]
    all_rootMass[ind_part_ts[ts_ind_attached]] = msub[
        cantorIDs[all_rootGal[ind_part_ts[ts_ind_attached]]]]

    if scale_root_lograd:
        extra_shid = yb.read_hdf5(cantorloc, 'Snapshot_' + str(isnap).zfill(3) 
                                  + '/Subhalo/Extra/SubhaloIndex') 
        cantor_galID = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(isnap).zfill(3) + '/Subhalo/Galaxy') 

        shmr_extra = yb.read_hdf5(cantorloc, 'Snapshot_' + str(isnap).zfill(3) 
                                  + '/Subhalo/Extra/Stars/'
                                  + 'QuantileRadii')[:, 1, 2]
        shmr_gal = np.zeros(numGal) - 1
        shmr_gal[cantor_galID[extra_shid]] = shmr_extra

        all_rootLograd[ind_part_ts[ts_ind_attached]] = (
            rad_gal[ts_ind_attached]/shmr_gal[ts_gal[ts_ind_attached]])
    else:
        all_rootLograd[ind_part_ts[ts_ind_attached]] = (
            np.log10(rad_gal[ts_ind_attached]))

    # Updating root snap is different in 'first' and 'last' mode:
    # 'first': increment snap of all NOT-matched particles
    # 'last' : set snap of all MATCHED particles to current
    if rootType == 'first':
        all_rootSnap[ind_part_ts[ts_ind_unmatched]] += 1
    else:
        all_rootSnap[ind_part_ts[ts_ind_attached]] = isnap

    # New bit added 2-Aug-19:
    # Optionally, parent particles that are unbound while still gas
    # 'forget' prior root associations:
    if rootType == 'last' and forgetful_parents:
        snapdir = st.form_files(rundir, isnap, 'snap')
        curr_aexp = st.snap_age(snapdir, 'aexp')
        ind_forgetting = np.nonzero(
            (all_ft > curr_aexp) & (ts_flag_attached == 0) &
            (all_rootGal >= 0))[0]

        all_rootGal[ind_forgetting] = -1
        all_rootMass[ind_forgetting] = -1
        all_rootLograd[ind_forgetting] = -1


    return    # ----- ends processing of current snapshot ----


def write_output(iiptype):
    """Write output for particle type iiptype"""

    pre = "PartType{:d}" .format(iiptype)

    pt_inds = np.arange(ptype_offset[iiptype], ptype_offset[iiptype+1])
    
    yb.write_hdf5(all_ids[pt_inds], outloc, pre+"/ParticleIDs", comment = "Particle IDs")
    yb.write_hdf5(all_rootSnap[pt_inds], outloc, pre+"/RootSnapshot", comment = "First snapshot in which a particle has been in a subhalo")
    yb.write_hdf5(all_rootGal[pt_inds], outloc, pre+"/RootGalaxy", comment = "Galaxy of particle in its root snapshot")

    if iiptype in [4, 5] and include_parents:
        pt_inds_parents = np.arange(ptype_offset[iiptype+2], ptype_offset[iiptype+3])
        yb.write_hdf5(all_rootSnap[pt_inds_parents], outloc, pre+"/ParentRootSnapshot", comment = "First snapshot in which the particle *or its gas parent particle* was in a subhalo.")
        yb.write_hdf5(all_rootGal[pt_inds_parents], outloc, pre+"/ParentRootGalaxy", comment = "Galaxy of particle or its gas parent parent in its root snapshot.")

    return

def get_particle_code(ig_rootGal, ig_hostGal):
    """
    Compute an origin code for each particle in a z=0 galaxy.

    This encodes how its root galaxy is related to its z=0 host:
      0: In-situ   (root galaxy = host)
      1: Accreted  (root galaxy merged with host)
      2: Quasi-merged (root galaxy alive, but M_star < threshold)
      3: Stripped  (root galaxy alive and M_star > threshold)
      4: Stolen    (root glaxy alive but in other FOF than host)
      5: Adopted   (root galaxy not alive, but did not merge with host)
      6: Tracefail (no root galaxy)

    Parameters:
    -----------
    ig_rootGal : ndarray (int)
        Root galaxy ID for all particles in a galaxy at z = 0.
    ig_hostGal : ndarray (int)
        z = 0 host galaxy for all particles in a galaxy at z = 0.

    Returns:
    --------
    code : ndarray (int8)
        Code between 0 and 5 that indicates its origin (see above).
    """

    # Read in central galaxy at z = 0 (to distinguish same/other FOF)
    cenGal_z0 = yb.read_hdf5(cantorloc, 'CenGalExtended')[:, snap_z0]

    # Read in stellar mass at z = 0 (to distinguish qm/stripped)
    mstar_z0 = yb.read_hdf5(cantorloc, 'Snapshot_' + str(snap_z0).zfill(3) + 
                            '/Subhalo/MassType')[:, 4]
    cid_z0 = cantor_shi[:, snap_z0]


    # ---------------------------------------------------
    # Decompose particles into the six origin categories:
    # ---------------------------------------------------

    print("Decompose particles into origin categories...")

    ind_tracefail = np.nonzero(ig_rootGal < 0)[0]
    
    # Break galaxies with root into in-/ex-situ:
    ind_insitu = np.nonzero(ig_rootGal == ig_hostGal)[0]
    ind_exsitu = np.nonzero((ig_rootGal >= 0) & (ig_rootGal != ig_hostGal))[0]

    # Break ex-situ down into different categories:
    subind_acc = np.nonzero(mergelist[ig_rootGal[ind_exsitu], snap_z0] ==
                            ig_hostGal[ind_exsitu])[0]
    subind_other = np.nonzero(mergelist[ig_rootGal[ind_exsitu], snap_z0] !=
                              ig_hostGal[ind_exsitu])[0]

    # --> Break 'ex-situ/other' down into 'root dead/alive at z = 0':
    ssubind_dead = np.nonzero(cid_z0[ig_rootGal[ind_exsitu[subind_other]]]
                              < 0)[0]
    ssubind_alive = np.nonzero(cid_z0[ig_rootGal[ind_exsitu[subind_other]]]
                               >= 0)[0]

    # ----> Break 'ex-situ/other/alive' down into 'in same/other FOF':
    ind_alive = ind_exsitu[subind_other[ssubind_alive]]
    sssubind_otherfof = np.nonzero(cenGal_z0[ig_rootGal[ind_alive]] != 
                                   cenGal_z0[ig_hostGal[ind_alive]])[0]
    sssubind_samefof = np.nonzero(cenGal_z0[ig_rootGal[ind_alive]] == 
                                  cenGal_z0[ig_hostGal[ind_alive]])[0]

    # ------> Break '.../alive/samefof' down into 'stripped/quasi-merged':
    if max_mstar_qm is None:
        ssssubind_qm = np.zeros(0, dtype = int)
        ssssubind_stripped = np.arange(len(sssubind_samefof))
    else:
        ssssubind_qm = np.nonzero(mstar_z0[cid_z0[ig_rootGal[ind_alive[
            sssubind_samefof]]]] <= 10.0**(max_mstar_qm-10))[0]
        ssssubind_stripped = np.nonzero(mstar_z0[cid_z0[ig_rootGal[ind_alive[
            sssubind_samefof]]]] > 10.0**(max_mstar_qm-10))[0]

    # ----------------- Now assign codes ----------------------------
    
    print("Assign origin codes to particles...")

    ig_code = np.zeros(len(ig_rootGal), dtype = np.int8) - 1

    ig_code[ind_tracefail] = 6
    ig_code[ind_insitu] = 0
    ig_code[ind_exsitu[subind_acc]] = 1
    ig_code[ind_exsitu[subind_other[ssubind_dead]]] = 5
    ig_code[ind_alive[sssubind_otherfof]] = 4
    ig_code[ind_alive[sssubind_samefof[ssssubind_stripped]]] = 3
    ig_code[ind_alive[sssubind_samefof[ssssubind_qm]]] = 2

    if np.min(ig_code < 0):
        print("Why are some galaxies not assigned a code?!?")
        set_trace()

    return ig_code


def bin_particles(all_ids, all_code, all_gal, all_rootMass, all_ft, 
                  all_rootLograd, ptype):
    """
    Form the total mass by origin code, host/root mass, host radius, and 
    environment.

    Parameters:
    -----------
    all_ids : ndarray(int)
        Particle IDs for all particles in a galaxy at z = 0.
    all_code : ndarray(int)
        Origin codes for particles
    all_gal : ndarray(int)
        The particles' galaxy at z = 0.
    """

    ip_off = ptype_offset[ptype]
    ip_end = ptype_offset[ptype+1]
    numPt = ip_end - ip_off


    # 1.) Extract particle masses for in-galaxy-at-z0 particles:

    pt_ids = all_ids[ip_off : ip_end]
    pt_code = all_code[ip_off : ip_end]
    pt_gal = all_gal[ip_off : ip_end]
    pt_rootMass = all_rootMass[ip_off : ip_end]
    pt_ft = all_ft[ip_off : ip_end]
    pt_rootLograd = all_rootLograd[ip_off : ip_end]

    pt_mass = np.zeros(numPt, dtype = np.float32)    
    
    if record_assembly and rootType == 'last':
        pt_tass = all_tass[ip_off : ip_end]
    

    if ptype > 5:
        ip_off_child = ptype_offset[ptype-2]
        ip_end_child = ptype_offset[ptype-1]
        pt_code_child = all_code[ip_off_child : ip_end_child]
        ptype_true = ptype - 2
    else:
        ptype_true = ptype

    pt_pre = 'PartType{:d}/' .format(ptype_true)
    snap_pt_ids = st.eagleread(snapdir_z0, pt_pre + 'ParticleIDs', 
                               astro = False)
    
    uniform_mass = False
    # Loading masses depends on particle type...
    if ptype_true == 1:
        pt_mass = np.zeros(numPt, dtype = np.float32)+st.m_dm(rundir)
        uniform_mass = True
    else:
        if use_initial_mass:
            if ptype_true == 0:
                pt_mass = np.zeros(numPt, dtype = np.float32)+st.m_bar(rundir)
                uniform_mass = True
            elif ptype_true == 4:
                snap_pt_mass = st.eagleread(
                    snapdir_z0, pt_pre + 'InitialMass', astro = True)[0]
            elif ptype_true == 5:
                snap_pt_mass = st.eagleread(
                    snapdir_z0, pt_pre + 'SubgridMass', astro = True)[0]
            else:
                print("Invalid particle type: {:d}" .format(ptype))
                set_trace()
        else:
            snap_pt_mass = st.eagleread(snapdir_z0, pt_pre + 'Mass', 
                                        astro = True)[0]
            if (ptype_true == 0 or ptype_true == 4) and use_metal_mass:
                snap_pt_zmet = st.eagleread(
                    snapdir_z0, pt_pre + 'SmoothedMetallicity', astro=True)[0]
                snap_pt_mass *= snap_pt_zmet

    # Now match to in-gal particles...
    
    if not uniform_mass:
        # Invert the *snapshot* list, and then fetch properties from there
        snap_revID = yb.create_reverse_list(snap_pt_ids)
        pt_snapInd = snap_revID[pt_ids]
        if len(pt_snapInd) == 0: set_trace()
        if np.min(pt_snapInd) < 0:
            print("Why could some particles not be matched??")
            set_trace()
    
        pt_mass = snap_pt_mass[pt_snapInd].astype(np.float32)

    # Special bit for parents: zero the mass for all particles whose
    # child is not in-situ (only care about these)
    if ptype > 5:
        ind_child_ex = np.nonzero(pt_code_child > 0)[0]
        pt_mass[ind_child_ex] = 0

    # 2.) Form bin codes for (a) host mass, (b) root mass, (c) host radius, 
    #     and (d) environment.

    snap_pre = 'Snapshot_' + str(snap_z0).zfill(3)
    pt_shi_z0 = cantor_shi[pt_gal, snap_z0]

    # a) host catalogue entry
    host_cat = catInd[pt_gal]
    if np.min(host_cat) < 0 or np.max(host_cat) >= n_host: 
        print("Invalid host index detected...")
        set_trace()

    pt_host_cat = host_cat.astype(np.int8)


    # b) root mass
    delta_m_root = (root_mass_range[1] 
                    - root_mass_range[0]) / (root_mass_nbins - 2)
    pt_root_mass = np.floor((pt_rootMass - root_mass_range[0]) 
                     / delta_m_root).astype(np.int8)
    pt_root_mass = np.clip(pt_root_mass, -1, root_mass_nbins-2) + 1

    # c) host radius
    ca_ids = yb.read_hdf5(cantorloc, snap_pre + '/IDs')
    ca_rad = yb.read_hdf5(cantorloc, snap_pre + '/Radius')
    ca_revID = yb.create_reverse_list(ca_ids)
    ca_inds = ca_revID[pt_ids]
    if np.min(ca_inds) < 0:
        print("Why could some particles not be matched into Cantor list?!")
        set_trace()

    pt_host_lograd = np.log10(ca_rad[ca_inds])

    # ... and bin up
    delta_r_host = (host_lograd_range[1] 
                    - host_lograd_range[0]) / (host_lograd_nbins - 2)
    pt_host_rad = np.floor((pt_host_lograd - host_lograd_range[0]) 
                    / delta_r_host).astype(np.int8)
    pt_host_rad = np.clip(pt_host_rad, -1, host_lograd_nbins-2) + 1

    
    # e) Root age:
    delta_age = (age_range[1] 
                 - age_range[0]) / (age_nbins)

    if record_assembly and rootType == 'last':
        pt_birthAge = pt_tass    # This is AGE OF THE UNIVERSE at assembly

    elif ptype_true == 4:
        pt_birthAge = csi_age(1/pt_ft-1)    # This is AGE OF THE UNIVERSE at formation

    else:
        pt_birthAge = np.zeros_like(delta_r_host)

    pt_age = np.floor((pt_birthAge - age_range[0])
              / delta_age).astype(np.int8)
        
    # f) Root radius:
    delta_root_lograd = (root_lograd_range[1]
                         - root_lograd_range[0]) / (root_lograd_nbins - 2)
    pt_root_lograd = np.floor((pt_rootLograd - root_lograd_range[0])
                      / delta_root_lograd).astype(int)
    pt_root_lograd = (np.clip(pt_root_lograd, -1, 
                              root_lograd_nbins-2)+1).astype(np.int8)

    # Fill dummy categories
    host_mhalo_nbins = 1
    host_relrad_nbins = 1
    pt_host_mhalo = np.zeros_like(pt_host_cat)
    pt_host_mhalo[:] = 0
    pt_host_relrad = np.copy(pt_host_mhalo)

    # 3.) Bin up masses with external C-routine
    
    print("Binning up masses... ", end = '', flush = True)

    mass_binned = np.zeros((7,                  # 7 possible origin codes
                            n_host,
                            host_mhalo_nbins,
                            root_mass_nbins,
                            host_lograd_nbins,
                            host_relrad_nbins,
                            age_nbins,
                            root_lograd_nbins), dtype = np.float64)


    # *********** IMPORTANT ********************************
    # This next line needs to be modified to point
    # to the full path of where the library has been copied.
    # *******************************************************
    
    ObjectFile = "/u/ybahe/ANALYSIS/PACKAGES/lib/sumbinsXD.so"

    c_numPart = c.c_long(numPt)
    c_nbins_code = c.c_byte(7)
    c_nbins_catHost = c.c_byte(n_host)
    c_nbins_massHalo = c.c_byte(host_mhalo_nbins)
    c_nbins_massRoot = c.c_byte(root_mass_nbins)
    c_nbins_radHost = c.c_byte(host_lograd_nbins)
    c_nbins_relradHost = c.c_byte(host_relrad_nbins) 
    c_nbins_age = c.c_byte(age_nbins)
    c_nbins_radRoot = c.c_byte(root_lograd_nbins)

    partMass_p = pt_mass.ctypes.data_as(c.c_void_p)

    code_p = pt_code.ctypes.data_as(c.c_void_p)
    hostCatBin_p = pt_host_cat.ctypes.data_as(c.c_void_p)
    hostMhaloBin_p = pt_host_mhalo.ctypes.data_as(c.c_void_p)
    rootMassBin_p = pt_root_mass.ctypes.data_as(c.c_void_p)
    hostRadBin_p = pt_host_rad.ctypes.data_as(c.c_void_p)
    hostRelradBin_p = pt_host_relrad.ctypes.data_as(c.c_void_p)
    rootRadBin_p = pt_root_lograd.ctypes.data_as(c.c_void_p)
    ageBin_p = pt_age.ctypes.data_as(c.c_void_p)


    result_p = mass_binned.ctypes.data_as(c.c_void_p)

    nargs = 19
    myargv = c.c_void_p * nargs
    argv = myargv(c.addressof(c_numPart), 
                  c.addressof(c_nbins_code),
                  c.addressof(c_nbins_catHost),
                  c.addressof(c_nbins_massHalo),
                  c.addressof(c_nbins_massRoot),
                  c.addressof(c_nbins_radHost),
                  c.addressof(c_nbins_relradHost),
                  c.addressof(c_nbins_age),
                  c.addressof(c_nbins_radRoot),
                  partMass_p, 
                  code_p,
                  hostCatBin_p, hostMhaloBin_p, rootMassBin_p, 
                  hostRadBin_p, hostRelradBin_p, ageBin_p, rootRadBin_p,
                  result_p)
    
    lib = c.cdll.LoadLibrary(ObjectFile)
    succ = lib.sumbins(nargs, argv)

    print("Sum of particle masses: ", np.sum(pt_mass))
    print("Sum of binned masses:   ", np.sum(mass_binned))


    return mass_binned


def write_header(outloc):
    """Write header information to HDF5 file."""

    yb.write_hdf5_attribute(outloc, 'Header', 'CantorCatalogue',
                            cantorCatalogue)
    yb.write_hdf5_attribute(outloc, 'Header', 'MinRatio', min_ratio)
    yb.write_hdf5_attribute(outloc, 'Header', 'CompType', comp_type)
    yb.write_hdf5_attribute(outloc, 'Header', 'IncludeParents', 
                            include_parents)
    yb.write_hdf5_attribute(outloc, 'Header', 'RootType', rootType)
    yb.write_hdf5_attribute(outloc, 'Header', 'MaxMstarQM', max_mstar_qm)
    
    yb.write_hdf5_attribute(outloc, 'Header', 'NHost', 
                            n_host)

    yb.write_hdf5_attribute(outloc, 'Header', 'RootMassRange', root_mass_range)
    yb.write_hdf5_attribute(outloc, 'Header', 'RootMassType', root_mass_type)
    yb.write_hdf5_attribute(outloc, 'Header', 'RootMassBins', root_mass_nbins)

    yb.write_hdf5_attribute(outloc, 'Header', 'HostLogradRange', 
                            host_lograd_range)
    yb.write_hdf5_attribute(outloc, 'Header', 'HostLogradBins', 
                            host_lograd_nbins)

    yb.write_hdf5_attribute(outloc, 'Header', 'HostRelradRange', 
                            host_relrad_range)
    yb.write_hdf5_attribute(outloc, 'Header', 'HostRelradBins', 
                            host_relrad_nbins)
    
    yb.write_hdf5_attribute(outloc, 'Header', 'InitialMass', use_initial_mass)
    yb.write_hdf5_attribute(outloc, 'Header', 'MetalMass', use_metal_mass)
    


# =======================================================================
# Actual program starts here
# =======================================================================
        
# Set up MPI to enable processing different sims in parallel
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

outloc = outloc + '.' + str(rank) + '.hdf5'

snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])

# Set up a fine interpolant to easily compute stellar ages:
zFine = np.arange(0, 20, 0.01)
ageFine = Planck13.age(zFine).value
csi_age = interp1d(zFine, ageFine, kind = 'cubic', fill_value = 'extrapolate')

# Load input catalogue
cat_sim = yb.read_hdf5(clusterCatalogue, 'Sim')
cat_gal = yb.read_hdf5(clusterCatalogue, 'Galaxy')
n_host = len(cat_sim)
print("There are {:d} galaxies in the input catalogue." .format(n_host))


if 0 in ptypeList:
    gas_mass_binned = np.zeros((n_sim,
                                7,                  # 7 possible origin codes
                                n_host,
                                1,
                                root_mass_nbins,
                                host_lograd_nbins,
                                1,
                                age_nbins,
                                root_lograd_nbins), dtype = np.float32)

if 1 in ptypeList:
    dm_mass_binned = np.zeros((n_sim,
                               7,                  # 7 possible origin codes
                               n_host,
                               1,
                               root_mass_nbins,
                               host_lograd_nbins,
                               1,
                               age_nbins,
                               root_lograd_nbins), dtype = np.float32)
    
if 4 in ptypeList:
    star_mass_binned = np.zeros((n_sim,
                                 7,                  # 7 possible origin codes
                                 n_host,
                                 1,
                                 root_mass_nbins,
                                 host_lograd_nbins,
                                 1,
                                 age_nbins,
                                 root_lograd_nbins), dtype = np.float32)

if 5 in ptypeList:
    bh_mass_binned = np.zeros((n_sim,
                               7,                  # 7 possible origin codes
                               n_host,
                               1,
                               root_mass_nbins,
                               host_lograd_nbins,
                               1,
                               age_nbins,
                               root_lograd_nbins), dtype = np.float32)

ptypeList_plus = ptypeList.copy()
if include_parents:
    if 4 in ptypeList:
        ptypeList_plus.append(6)
    if 5 in ptypeList:
        ptypeList_plus.append(7)

if 6 in ptypeList_plus:
    starsParent_mass_binned = np.zeros((n_sim,
                                        7,                  # 7 possible origin codes
                                        n_host,
                                        1,
                                        root_mass_nbins,
                                        host_lograd_nbins,
                                        1,
                                        age_nbins,
                                        root_lograd_nbins), dtype = np.float32)
    



if not os.path.exists(yb.dir(outloc)):
    os.makedirs(yb.dir(outloc))

if os.path.exists(outloc):
    os.rename(outloc, outloc + '.old')


write_header(outloc)

for isim in range(n_sim):

    # Skip this one if we are multi-threading and it's not for this task to 
    # worry about
    if not isim % numtasks == rank:
        continue
        
    sim_stime = time.time()

    print("")
    print("**************************")
    print("Now processing halo CE-{:d}" .format(isim))
    print("**************************")
    print("")

    sys.stdout.flush()

    # Set up standard path and file names:
    if simname == "Eagle":
        rundir = basedir
        hldir = er.clone_dir(rundir) + '/highlev/'
        fgtloc = hldir + '/FullGalaxyTables.hdf5'
    else:
        rundir = basedir + '/CE-' + str(isim) + '/' + runtype
        hldir = rundir + '/highlev'
        fgtloc = hldir + '/FullGalaxyTables.hdf5' 
        spiderloc = hldir + '/SpiderwebTables.hdf5'

    if not os.path.exists(rundir):
        print("Can not find rundir...")
        continue

    # Find which galaxies need processing from this sim
    cat_thissim = np.nonzero(cat_sim == isim)
    if len(cat_thissim) == 0:
        print("No galaxies required from simulation {:d}, exiting..."
              .format(isim))
        continue

    else:
        print("Simulation {:d} contains {:d} target galaxies."
              .format(isim, len(cat_thissim)))
        cat_gal_thissim = cat_gal[cat_thissim]


    # Set up files to load particles from.
    snapdir_z0 = st.form_files(rundir, snap_z0, 'snap')
    snapdir_0 = st.form_files(rundir, 0, 'snap')

    aexp_z0 = st.snap_age(snapdir_z0, type = 'zred')
    lbt_now = Planck13.lookback_time(1/aexp_z0-1).value 
    age_now = Planck13.age(1/aexp_z0-1).value 

    catID = calendar.timegm(time.gmtime())
    yb.write_hdf5_attribute(outloc, "Header", "CatalogueID", catID)

    sw = st.Spiderweb(hldir, highlev = True)

    cantorloc = (ht.clone_dir(hldir) 
                 + '/CantorCatalogue.hdf5')
    """
    if not os.path.exists(cantorloc):
        cantorloc = (ht.clone_dir(hldir)
                     + '/CantorCatalogue_26Jun19.hdf5')
    if not os.path.exists(cantorloc):
        cantorloc = (ht.clone_dir(hldir)
                     + '/CantorCatalogue_27Jun19.hdf5')
    if not os.path.exists(cantorloc):
        cantorloc = (ht.clone_dir(hldir)
                     + '/CantorCatalogue_28Jun19.hdf5')
    """ 
    if not os.path.exists(cantorloc):
        print("Cannot find cantor catalogue...")
        print(cantorloc)
        continue

    cantor_shi = yb.read_hdf5(cantorloc, 'SubhaloIndex')
    
    # Set up translation table to associate host galaxy --> cat entry
    numGal = cantor_shi.shape[0]
    catInd = np.zeros(numGal, dtype = int)-1
    catInd[cat_gal_thissim] = cat_thissim
            
    # Read in central galaxy at z = 0 (to distinguish same/other FOF)
    cenGal_z0 = yb.read_hdf5(cantorloc, 'CenGalExtended')[:, snap_z0]
    
    mergelist = yb.read_hdf5(spiderloc, 'MergeList')
    contFlag = yb.read_hdf5(fgtloc, 'Full/ContFlag')[:, 1]

    # Load required information from galaxy evolution tables:
    if min_ratio is not None:
        #msub_all = yb.read_hdf5(fgtloc, 'Msub')
        gal_sh_z0 = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' 
                                 + str(snap_z0).zfill(3) + '/Galaxy')

    print("Done with setup, now loading particles...")
    sys.stdout.flush()
 
    # Load particle IDs for in-galaxy-at-z=0 particles:
    all_ids, all_gal_z0, all_ft, ptype_offset = load_particles()
    
    maxid = np.max(all_ids)
    numPart = len(all_ids)

    if min_ratio is not None:
        # Set up the `comparison mass' array. This is the `ask mass'
        # that a potential host needs to have (within min_ratio) in order to 
        # be able to 'adopt' the particle. 
        
        all_compMass = np.zeros(numPart, dtype = np.float32)

        # If we compare at z = 0, can already fill comparison array now:
        # (with 'current', masses will be loaded/updated in each snapshot).
        if comp_type == 'z0':
            
            msub_z0 = yb.read_hdf5(cantorloc, 'Snapshot_' + 
                                   str(snap_z0).zfill(3) + '/Subhalo/Mass')
            
            all_compMass = msub_z0[cantor_shi[all_gal_z0, snap_z0]]


    # Back in 'general' mode. Find the root snapshot for each particle
    # (first to search)

    all_rootSnap = np.zeros(numPart, dtype = np.int8)
    if rootType == 'last':
        all_rootSnap += 30

    all_rootGal = np.zeros(numPart, dtype = np.int32)-1
    all_rootMass = np.zeros(numPart, dtype = np.float32)-1
    all_rootLograd = np.zeros(numPart, dtype = np.float32)+1000
    if record_assembly:
        all_tass = np.zeros(numPart, dtype = np.float32) - 1

    # In this version, we don't need to initialize root-snaps, because
    # non-parent types will only look in their own type.

    """
    for iiptype in ptypeList:
        if iiptype in [4, 5]:
            all_rootSnap[ptype_offset[iiptype]:ptype_offset[iiptype+1]] = (
                find_root_snap(iiptype))
    """ 

    print("Done loading particles, finding root galaxies...")

    for isnap in range(nsnap):
        process_snapshot(rundir, isnap, cantorloc)

    print("")
    print("Now determining origin code for all particles...")
    

    # Determine 'type-code' for all (in-z0-gal) particles:
    all_code = get_particle_code(all_rootGal, all_gal_z0)

    #yb.write_hdf5(all_code, partloc, 'Code', new = True)
    #yb.write_hdf5(all_ids, partloc, 'ID')

    print("Binning up particle masses for each type...")

    for ptype in ptypeList_plus:

        mass_binned = bin_particles(all_ids, all_code, all_gal_z0, 
                                    all_rootMass, all_ft, all_rootLograd, 
                                    ptype)    

        # Now combine results across simulations and write
        if ptype == 0:
            gas_mass_binned[isim, ...] = mass_binned.astype(np.float32)
            yb.write_hdf5(gas_mass_binned, outloc, 'BinnedMasses_Gas')
        elif ptype == 1:
            dm_mass_binned[isim, ...] = mass_binned.astype(np.float32)
            yb.write_hdf5(dm_mass_binned, outloc, 'BinnedMasses_DM')
        elif ptype == 4:
            star_mass_binned[isim, ...] = mass_binned.astype(np.float32)
            yb.write_hdf5(star_mass_binned, outloc, 'BinnedMasses_Stars')
        elif ptype == 6:
            starsParent_mass_binned[isim, ...] = mass_binned.astype(np.float32)
            yb.write_hdf5(starsParent_mass_binned, outloc, 
                          'BinnedMasses_StarsParent')



print("Done!")
