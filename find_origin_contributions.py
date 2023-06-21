"""
Extract "birth galaxy" of all stellar (and optionally other) particles.

Adapted version that computes the mass contributed by the top 30 birth
galaxies in each radial bin for each host (started 1-Aug-2019).

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

 -- Updated 1-Aug-19:  adapted to compute mass contribution of most
                       contributing galaxies

 -- Update starting 19-Jun-23: trying to make this the universal extraction
                               script for all purposes. Also tidy up. 

"""

import numpy as np
from astropy.io import ascii
from pdb import set_trace
import calendar
import time
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("Not using MPI...")
    MPI = None
import os
import hydrangea as hy
import sys
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13

from hydrangea import hdf5 as hd


simname = "Hydrangea"     # 'Eagle' or 'Hydrangea'
runtype = "HYDRO"         # 'HYDRO' or 'DM'
ptypeList = [4]           # List of particle types to process

# File name of the Cantor Catalogue to use:
cantorCatalogue = 'CantorCatalogue.hdf5' 

# File name of *cluster* catalogue to use as input:
clusterCatalogue = ('../cluster_catalogue_z0_22Jul19_merge1p10_wSFR.hdf5')

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

# Specify whether we want to attach particles to the first galaxy to which
# they are associated, or to the last one to which they belong before they
# join their final (z = 0) galaxy.
rootType = 'last'  # 'first' (first SH association) or 'last' (before final) 

record_assembly = False

# --------- Options for origin code determination -------------

# Threshold between merged and quasi-stripped, in log (M_star/M_sun)
# (set to None to not apply any threshold, then no quasi-stripped):
max_mstar_qm = 9.0   


# --------- Options for output catalogue ----------------------

# Minimum and maximum mass of root galaxy bins:
root_mass_range = [8.0, 13.0]
root_mass_type = 'msub'
root_mass_nbins = 3   # incl. 2 for outliers

# Min and max radius in host (in log Mpc), and number of bins:
host_lograd_range = [-3, 0.5]
host_lograd_nbins = 7   # incl. 2 for outliers

# Min and max rel-rad, and number of bins:
host_relrad_range = [0, 10]
host_relrad_nbins = 2     # incl. 1 for outliers (no negative relRad!)

# Min and max age (in Gyr), and number of bins:
age_range = [0, 14]
age_nbins = 1     # no outliers here

# Min and max birth radius (in log Mpc), and number of bins:
root_lograd_range = [-4, 0.5]
root_lograd_nbins = 3     # incl. 2 outliers

assembly_lograd_range = [-3, 0.5]
assembly_lograd_nbins = 3

# Scale root radius by stellar half mass radius?
scale_root_lograd = False

# Use initial mass for stars (and initial baryon mass for gas)? If False,
# the mass at z = 0 will be used.
use_initial_mass = False  
 
# Count only metal mass (False: total stellar mass)?
use_metal_mass = False

# Limit parent tracing to x Gyr prior to star formation time?
parent_limit = None #2.0

# Set the level at which galaxies have to be uncontaminated to count as roots
contFlagLevel = 1

# Specify how to bin the output. Options are "origin" (for origin codes) or
# "topgal" for N most contributing galaxies
bin_type = 'origin'
numOriginGalaxies = 3000


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

ptype_names = ['Gas', 'DM', '', '', 'Stars', 'BHs']

outloc = ('../'
          'OriginMasses_Test')

# =============================================================================


def find_galaxy_for_ids(sim, ids, isnap, ptype=None, ptype_offset=None,
                        check_cont=False, return_rad=False):
    """
    Identify the galaxies that host input particle IDs in a given snapshot.

    The association is done self-consistently based on the Cantor catalogues.
    The `ptype` and `ptype_offset` parameters offer the option to only
    consider particles that are of a certain type in this snapshot, which is
    typically useful to exclude the gas progenitors of star particles.

    Parameters
    ----------
    sim : Simulation instance
        The simulation that is being processed.
    ids : ndarray
        The input array of IDs whose galaxies should be found.
    isnap : int
        The snapshot in which galaxies should be found.
    ptype : int or None (default)
        The particle type index to consider exclusively, if it is the same for
        all input IDs. If None, all particle types are considered for all IDs,
        which means that e.g. a galaxy may be identified for a gas progenitor
        of a star particle.
    ptype_offset : list(7) or None (default)
        A list encoding the element type as which each particle should be
        looked up. Entries 0-5 specify the first element of each particle type
        in the input ID list. Entry 6 specifies the beyond-last element
        for type 5 (BHs). For all IDs from entry 6 onwards, all particle types
        are considered.
    check_cont : bool, optional
        Explicitly check whether the identified host galaxies are flagged
        as contaminated (depending on the level of contflag). If this is the
        case, the particles are not attached to their galaxies. Default: False
    return_rad : bool, optional
        Look up and return the radius of each particle within its galaxy.
        Default: False

    Returns
    -------
    galaxies : ndarray
        An array of the same shape as `ids` that contains the galaxy to which
        each particle belongs in snapshot `isnap`. -1 is used as filler value
        for any particles that are not in any (eligible) galaxy.
    ind_in_gal : ndarray
        The indices of particles (into ids/galaxies) that are actually found
        to be in a galaxy.
    rad_gal : ndarray, optional
        The radii of each particle within its galaxy, ?in pMpc?. 1000 is used
        as a filler value for particles that are not associated to a galaxy.
        Only returned if `return_rad` is True.

    Note
    ----
    We use the prefix `ca_` for many variables here to indicate that these
    refer to the Cantor tables.
    """

    # Form the Cantor files for this snapshot
    cantorloc = f'{sim.high_level_dir}/Cantor/Cantor_{isnap:03d}.hdf5'
    cantoridloc = f'{sim.high_level_dir}/Cantor/Cantor_{isnap:03d}_IDs.hdf5'
    cantorradloc = f'{sim.high_level_dir}/Cantor/Cantor_{isnap:03d}_Radii.hdf5'

    # Load this snapshot's Cantor particle data. We load both the unsplit and
    # split by particle type versions of the offset lists.
    ca_ids = hd.read_data(cantoridloc, 'IDs')
    ca_offset = hd.read_data(cantorloc, 'Subhalo/Offset')
    ca_length = hd.read_data(cantorloc, 'Subhalo/Length')
    ca_offset_type = hd.read_data(cantorloc, 'Subhalo/OffsetType')
    ca_shi_gal = hd.read_data(cantorloc, 'Subhalo/Galaxy')

    # The offset_type list has a coda in the second index (ptypes), so that we
    # can just subtract subsequent offsets to get the corresponding lengths
    ca_length_type = ca_offset_type[:, 1:] - ca_offset_type[:, :-1]

    # First step: locate all input IDs in the Cantor ID list. We do this for
    # all particles at once for better efficiency. The implementation deals
    # with particles outside the Cantor ID range. Although this also returns
    # a separate `ind_found_ca` list, this is not used below (`ca_inds` < 0
    # is interpreted as not found).
    ca_inds, ind_found_ca = hy.crossref.find_id_indices(ids, ca_ids)

    # Second step: convert `ca_inds` into Cantor subhalo indices (ca_shi).
    # Again, we use -1 as a filler value.
    ca_shi = np.zeros(len(ids), dtype = np.int32)-1
    
    # First mode: consider only particles of one (real) type.
    if ptype is not None:
        ca_shi = hy.ind_to_block(
            ca_inds, ca_offset_type[:, ptype], ca_length_type[:, ptype])

    # Second mode: combined particle list, treat different types separately.
    # Again quite straightforward, but we have to consider each type in turn.
    elif ptype_offset is not None:
        for iptype in range(6):
            if ptype_offset[iptype + 1] > ptype_offset[iptype]:
                start_pt = ptype_offset[iptype]
                end_pt = ptype_offset[iptype+1]
                ca_shi[start_pt:end_pt] = (
                    hy.ind_to_block(
                        ca_inds[start_pt:end_pt],
                        ca_offset_type[:, iptype],
                        ca_length_type[:, iptype]
                    )
                )
    
        # ... and special consideration for "fake" type 6: consider all
        # particle types within a galaxy (typically for parent finding).
        # Note that we use the unsplit offset/length lists here.
        ca_shi[ptype_offset[6]:] = (
            hy.ind_to_block(
                ca_inds[ptype_offset[6]:], ca_offset, ca_length)
        )

    # Third mode: no particle types, consider all entries for all galaxies.
    # This is the most straightforward case of all.
    else:
        ca_shi = hy.ind_to_block(ca_inds, ca_offset, ca_length)

    # Last bit is mode-independent: find particles in subhaloes, and 
    # convert SHI --> galaxy for these. If desired, artificially treat
    # particles in contaminated galaxies as not in a galaxy.
    ind_in_gal = np.nonzero(ca_shi >= 0)[0]
    if check_cont:
        subind_clean = np.nonzero(contFlag[ca_shi_gal[ca_shi[ind_in_gal]]] == 0)[0]
        ind_in_gal = ind_in_gal[subind_clean]

    ca_gal = np.zeros_like(ca_shi) - 1
    ca_gal[ind_in_gal] = ca_shi_gal[ca_shi[ind_in_gal]]

    return_list = [ca_gal, ind_in_gal]

    # Optional extra: read particle radii from Cantor catalogue too
    # (since they use the same alignment as IDs)
    if return_rad:
        rad = hd.read_data(cantorradloc, 'Radius')
        rad_gal = np.zeros(ca_gal.shape[0], dtype=np.float32)+1000
        rad_gal[ind_in_gal] = rad[ca_inds[ind_in_gal]]
        return_list.append(rad_gal)

    return return_list
    

def load_particles(sim, include_parents=True):
    """
    Load all particles of desired type(s) in galaxies at z = 0.

    The type(s) to be loaded are specified by the global ptypeList.
    For stars/BHs, parent gas particles can be included as duplicate,
    with ptype = 6/7 (i.e. real ptype + 2).

    Parameters
    ----------
    sim : Simulation instance
        The simulation for which to load particles.
    include_parents : bool, optional
        Switch to enable duplicating star/BH particles for their gas parents
        (default: True)

    Returns:
    --------
    all_ids : ndarray (int)
        The IDs of particles to process
    all_gal : ndarray (int)
        The galaxy to which each particle belongs at z = 0.
    all_ft : ndarray (float)
        The formation time of each particle (0 for all but stars/BHs).
    ptype_offset : ndarray (int) [9]
        The offsets of individual particle types in the ID list.
    """ 

    all_ids = np.zeros(0, dtype=int)
    all_gal = np.zeros(0, dtype=int)
    all_ages = np.zeros(0, dtype=float)
    all_logradii = np.zeros(0, dtype=float)
    ptype_offset = np.zeros(9, dtype=int)

    #cantor_dir = f'{sim.high_level_dir}/Cantor'
    #cantorloc = f'{cantor_dir}/Cantor_{snap_z0:03d}.hdf5'
    #cantoridloc = f'{cantor_dir}/Cantor_{snap_z0:03d}_IDs.hdf5'
    #cantorradloc = f'{cantor_dir}/Cantor_{snap_z0:03d}_Radii.hdf5'

    #ca_logradii = np.log10(hd.read_data(cantorradloc, 'Radius'))
    #ca_ids = hd.read_data(cantoridloc, 'IDs')
    #ca_revIDs = hy.crossref.ReverseList(ca_ids)
    
    # Loop through individual (to-be-considered) particle types:
    for iiptype in ptypeList:    
        print(f"   ... ptype {iiptype}...")

        # Load all IDs at z = 0 and identify their z = 0 host galaxy
        ptype = hy.SplitFile(
            sim.get_snap_file(snap_z0), part_type=iiptype, verbose=0)
        ptype_gal, ptype_in_gal, rad_gal = find_galaxy_for_ids(
            sim, ptype.ParticleIDs, snap_z0, ptype=iiptype, check_cont=True,
            return_rad=True
        )

        # Only keep particles that are in one of the input galaxies.
        # (`catInd` is non-negative only for input galaxies)
        subind_in_cat = np.nonzero(catInd[ptype_gal[ptype_in_gal]] >= 0)[0]
        ptype_in_cat = ptype_in_gal[subind_in_cat]

        # Now append matches to full list (across ptypes). We must cast 
        # .ParticleID explicitly to int, because the concatenation otherwise
        # converts to float...
        all_ids = np.concatenate(
            (all_ids, ptype.ParticleIDs[ptype_in_cat].astype(int)))
        all_gal = np.concatenate((all_gal, ptype_gal[ptype_in_cat]))
        ptype_offset[iiptype+1:] += len(ptype_in_cat)

        all_logradii = np.concatenate(
            (all_logradii, np.log10(rad_gal[ptype_in_cat])))
        
        # Also need to record the formation time of each particle.
        if iiptype == 4:
            ptype_ft = ptype.StellarFormationTime
        elif iiptype == 5:
            ptype_ft = ptype.BH_FormationTime
        else:
            ptype_ft = np.zeros(len(ptype_gal))
        ptype_ages = csi_age(ptype_ft)
        all_ages = np.concatenate((all_ages, ptype_ages[ptype_in_cat]))

    # If parent-finding is included, we simply duplicate the required IDs.
    # That way, we don't have to include parents as separate category in the 
    # main parts of the program.
    if include_parents:
        for iiParType in [4, 5]:
            if iiParType in ptypeList: 
                all_ids = np.concatenate((
                    all_ids, 
                    all_ids[ptype_offset[iiParType]:ptype_offset[iiParType+1]]
                ))
                all_gal = np.concatenate((
                    all_gal,
                    all_gal[ptype_offset[iiParType]:ptype_offset[iiParType+1]]
                ))
                all_ages = np.concatenate((
                    all_ages,
                    all_ages[ptype_offset[iiParType]:ptype_offset[iiParType+1]]
                ))

                # Increase offsets for last `types' (stored beyond pt-5):    
                ptype_offset[iiParType+3:] += (
                    ptype_offset[iiParType+1] - ptype_offset[iiParType])
           
    data = {}
    data['NumberOfParticles'] = len(all_ids)
    data['IDs'] = all_ids
    data['z0_Galaxies'] = all_gal
    data['FormationTimes'] = all_ages
    data['LogRadii'] = all_logradii
    data['PtypeOffsets'] = ptype_offset
    print(f"   ... loaded {len(all_ids)}...")
    return data


def process_snapshot(isnap, sim, root_data, particle_data, cantor_shi):
    """Find root galaxies of the particle set in one snapshot (isnap).

    Parameters
    ----------
    isnap : int
        The snapshot to process
    sim : Simulation instance
        The simulation on which we are working
    root_data : dict
        The under-assembly dict of root properties, will be updated.  
    cantor_shi : ndarray
        The Cantor galaxy --> SHI table for all snapshots.

    Note
    ----
    The prefix/suffix "ts" in the code refers to "this snapshot".

    """
    print("")
    print(f"---- Starting snapshot = {isnap} (sim = {sim.isim}) ------ ")
    print("")
    sys.stdout.flush()

    root_snaps = root_data['snapshots']
    root_galaxies = root_data['galaxies']
    root_masses = root_data['masses']
    root_logradii = root_data['logradii']
    all_tass = root_data['assembly_times']
    all_rass = root_data['assembly_radii']

    all_ids = particle_data['IDs']
    all_ages = particle_data['FormationTimes']
    ptype_offset = particle_data['PtypeOffsets']
    all_gal_z0 = particle_data['z0_Galaxies']

    # Need to consider all particles that have not yet been marked as found.
    # This is different depending on rootType: if "first", we only consider
    # those that are assigned to be processed in this snapshot, otherwise,
    # consider all (since we may update root galaxies).
    if rootType == 'first':
        ind_part_ts = np.nonzero(root_snaps == isnap)[0]
    else:
        if parent_limit is None:
            ind_part_ts = np.arange(len(root_snaps))
        else:
            curr_time = snap_ages[isnap]
            ind_part_ts = np.nonzero(all_ages < curr_time + parent_limit)[0]

    np_snap = len(ind_part_ts)
    if np_snap == 0:
        return root_data
    print(f"Processing {np_snap} particles in snapshot {isnap}...")

    # Extract current gal-->cantorID list, for simplicity:
    cantorIDs = cantor_shi[:, isnap]

    # ** Key part: **
    # Identify the galaxy (if any) of all currently-considered particles
    ts_gal, ts_ind_in_gal, ts_rad_gal = find_galaxy_for_ids(
        sim, all_ids[ind_part_ts], isnap, ptype_offset=ptype_offset,
        return_rad=True
    )
    np_gal = len(ts_ind_in_gal)
    print(f"... of which {np_gal} are in a galaxy...")
    if np_gal == 0: return root_data

    # Now check which identifications are 'valid', which depends on the 
    # program settings (prefix 'ig' --> 'in_gal'), and also requires knowing
    # the mass (total or stellar) of the subhaloes
    cantorloc = f'{sim.high_level_dir}/Cantor/Cantor_{isnap:03d}.hdf5'
    if root_mass_type == 'msub':
        msub = hd.read_data(cantorloc, 'Subhalo/Mass') * 1e10
    elif root_mass_type == 'mstar':
        msub = hd.read_data(cantorloc, 'Subhalo/MassType')[:, 4] * 1e10
    else:
        raise ValueError(f"Invalid root mass type '{root_mass_type}'")
    msub = np.log10(msub)

    if rootType == 'first':
        # Simple case: attach each particle only to the first available galaxy
        # (in this case we only test not-yet-attached particles anyway):
        subind_permitted = np.arange(np_gal)
 
    else:
        # In this case, we need to check whether particles can be attached
        # to their current host.        
        # Use prefix 'tsg' --> this snap, in gal. Recall, `ind_part_ts` is
        # the index of currently tested particles into the full list.
        tsg_gal = ts_gal[ts_ind_in_gal]
        tsg_old_root = root_galaxies[ind_part_ts[ts_ind_in_gal]]
        tsg_z0gal = all_gal_z0[ind_part_ts[ts_ind_in_gal]]
        tsg_msub = msub[cantorIDs[tsg_gal]]
        tsg_root_mass = root_masses[ind_part_ts[ts_ind_in_gal]]
        if record_assembly:
            tsg_tass = all_tass[ind_part_ts[ts_ind_in_gal]]
            
        if rootType == 'strictly_last':
            # Also simple: (re-)attach anything that isn't in the z0 host
            subind_permitted = np.nonzero(tsg_old_root != tsg_z0gal)[0]
        else:
            # a) Not yet attached --> always attach to current host
            # b) Attached to z=0 host --> only attach if to z=0 host
            # c) Attached to other gal --> only attach is NOT to z=0 host
            # 
            # The logic below is as follows: permitted are particles that
            # (i) don't have a root yet, or if they do
            # (ii-a) are now in a more massive galaxy than its current root AND
            # (ii-b-1) are attached to and remain in the z=0 galaxy, OR
            # (ii-b-2) remain not attached to z = 0 gal.
            # This allows finding the most massive progenitor, while avoiding
            # complications for backsplash galaxies.
            subind_permitted = np.nonzero(
                (tsg_old_root < 0) | 
                (
                    (tsg_msub > tsg_root_mass) &
                    (
                        (tsg_old_root == tsg_z0gal) & (tsg_gal == tsg_z0gal)
                    ) |
                    (
                        (tsg_old_root != tsg_z0gal) & (tsg_gal != tsg_z0gal)
                    )
                )
            )[0]

        # Record the assembly time for particles that have just assembled
        if record_assembly:
            curr_time = snap_ages[isnap]
            
            # Find particles that have just assembled (in z0 host, not before).
            # We do not need to take the 'permitted' mask into account here,
            # because we are now interested in anything that has just joined
            # the z = 0 host.
            subind_assembled = np.nonzero(
                (tsg_tass < 0) & (tsg_gal == tsg_z0gal))[0]
            ind_assembled = ind_part_ts[ts_in_in_gal[subind_assembled]]
            
            all_tass[ind_assembled] = curr_time

            lograd_ass = np.log10(ts_rad_gal[ts_ind_in_gal[ind_assembled]])
            all_rass[ind_assembled] = lograd_ass
            set_trace()

    # For convenience: direct index of permitted into this-snap particles
    ts_permitted = ts_ind_in_gal[subind_permitted]

    # Second criterion: is the host massive enough to count? Recall that
    # min_ratio is the ratio of host mass to z = 0 mass

    # Simple case: no minimum -- any host will do:
    if min_ratio is None:
        subind_massive = np.arange(len(subind_permitted))

    # Not-so-simple case: need to check which hosts are massive enough...
    else:
        # Two ways of testing 'massive enough': 
        # (i) Based on current masses
        if comp_type == 'current':

            # Need to first find comparison (`ask') mass of z = 0 host.
            # We initialize this to zero, which means that any particle whose
            # host is not alive at current snapshot is automatically eligible
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
                msub[cantorIDs[ts_gal[ts_permitted]]]
                >= min_ratio * all_compMass[ind_part_ts[ts_permitted]]
            )[0]

        # (ii) Based on z = 0 masses (this is simpler -- already loaded)
        else:
            subind_massive = np.nonzero(
                msub_z0[cantor_shi[ts_gal[ts_permitted], snap_z0]] 
                >= min_ratio * all_compMass[ind_part_ts[ts_permitted]]
            )[0]

    # Almost done! 
    ts_ind_attached = ts_permitted[subind_massive]
    ind_attached = ind_part_ts[ts_ind_attached]

    # Set up a mask to record which particles could be matched
    # (0 --> no, 1 --> yes)
    ts_flag_attached = np.zeros(np_snap, dtype = np.int8)
    ts_flag_attached[ts_ind_attached] = 1
    ts_ind_unmatched = np.nonzero(ts_flag_attached == 0)[0]
    print(f"Could match {len(subind_massive)} particles to galaxies "
          f"({len(ts_ind_unmatched)} unmatched).", flush=True)

    # Update root galaxy and mass for (re-)attached galaxies:
    root_galaxies[ind_attached] = ts_gal[ts_ind_attached]
    root_masses[ind_attached] = msub[cantorIDs[root_galaxies[ind_attached]]]

    if scale_root_lograd:
        extra_shid = hd.read_data(cantorloc, 'Subhalo/Extra/SubhaloIndex')
        cantor_galID = hd.read_data(cantorloc, 'Subhalo/Galaxy')
        shmr_extra = hd.read_data(
            cantorloc, 'Subhalo/Extra/Stars/QuantileRadii')[:, 1, 2]
        shmr_gal = np.zeros(cantor_shi.shape[0]) - 1
        shmr_gal[cantor_galID[extra_shid]] = shmr_extra
        root_logradii[ind_attached] = (
            ts_rad_gal[ts_ind_attached] / shmr_gal[ts_gal[ts_ind_attached]])
    else:
        root_logradii[ind_attached] = np.log10(ts_rad_gal[ts_ind_attached])

    # Updating root snap is different in 'first' and 'last' mode, to match
    # the different identifications of eligible particles.
    # 'first': increment snap of all NOT-matched particles by one, so they
    #          will be tested in the next snapshot; matched ones remain
    #          at this snapshot index.
    # 'last':  set snap of all MATCHED particles to current. All particles
    #          will be re-tested in subsequent snapshots anyway.
    if rootType == 'first':
        root_snaps[ind_part_ts[ts_ind_unmatched]] += 1
    else:
        root_snaps[ind_part_ts[ts_ind_attached]] = isnap

    # New bit added 2-Aug-19:
    # Optionally, parent particles that are unbound while still gas
    # 'forget' prior root associations:
    if rootType == 'last' and forgetful_parents:
        curr_time = snap_ages[isnap]
        ind_forgetting = np.nonzero(
            (all_ages > curr_time) & (ts_flag_attached == 0) &
            (root_galaxies >= 0))[0]
        root_galaxies[ind_forgetting] = -1
        root_masses[ind_forgetting] = -1
        root_logradii[ind_forgetting] = -1

    # ============= Processing of current snapshot is finished =============

    return root_data


class SplitList:
    """Class to simplify particle lookup by a given property."""

    def __init__(self, quant, lims):
        """
        Class constructur.

        Parameters:
        -----------
        quant : ndarray
            The quantity by which elements should be retrievable.
        lims : ndarray
            The boundaries (in the same quantity and units as quant) of 
            each retrievable 'element'.
        """
        
        self.argsort = np.argsort(quant)
        self.splits = np.searchsorted(quant, lims, sorter = self.argsort)

    def __call__(self, index):
        """
        Return all input elements that fall in a given bin.

        Parameters:
        -----------
        index : int
            The index corresponding to the supplied 'lims' array for which
            elements should be retrieved. All elements with 'quant' between
            lims[index] and lims[index+1] will be returned.
        
        Returns:
        --------
        elements : ndarray
            The elements that lie in the desired quantity range.
        """

        return self.argsort[self.splits[index]:self.splits[index+1]]  


def bin_particles(
        sim, particle_data, root_data, all_code, ptype, cat_gal_thissim):
    """
    Bin up the particles, either by origin code or by contributing galaxies.

    Parameters:
    -----------
    particle_data : dict
        Input particle data structure
    root_data : dict
        Structure with root galaxy data for each particle
    all_code : ndarray(int)
        The origin code of each particle
    ptype : int
        The particle type to be considered
    cat_gal_thissim : ndarray(int)
        target galaxies processed in this simulation

    Returns:
    --------
    top_mass : ndarray (float) [n_host, host_lograd_nbins, numOriginGalaxies]
        For each host and rad bin, the total mass contributed by the 
        N most contributing galaxies (in descending order), and then 
        everything else in the final bin.
    """

    # Extract relevant fields from particle and host dicts
    all_ids = particle_data['IDs']
    all_hostGal = particle_data['z0_Galaxies']
    all_hostLogRad = particle_data['LogRadii']
    if record_assembly:
        all_ages = root_data['assembly_times']
        all_assemblyLogRad = root_data['assembly_lograd']
    else:
        all_ages = particle_data['FormationTimes']
        all_assemblyLogRad = np.zeros_like(all_ages)
    ptype_offset = particle_data['PtypeOffsets']

    all_rootMass = root_data['masses']
    all_rootGal = root_data['galaxies']
    all_rootLogRad = root_data['logradii']
    max_root_gal = np.max(all_rootGal)

    # Find range of particles to consider here. Recall that we don't have
    # parents in this mode, so can use 'simple' ptype_offset list.
    ip_off = ptype_offset[ptype]
    ip_end = ptype_offset[ptype+1]
    numPt = ip_end - ip_off

    # Extract the particles that are of current ptype
    pt_ids = all_ids[ip_off : ip_end]
    pt_rootGal = all_rootGal[ip_off : ip_end]
    pt_rootMass = all_rootMass[ip_off : ip_end]
    pt_hostGal = all_hostGal[ip_off : ip_end]
    pt_code = all_code[ip_off : ip_end]
    pt_ages = all_ages[ip_off : ip_end]
    pt_rootLograd = all_rootLogRad[ip_off : ip_end]
    pt_hostLograd = all_hostLogRad[ip_off : ip_end]
    pt_assemblyLograd = all_assemblyLogRad[ip_off : ip_end]

    # ----------------------------------------------------------------------
    # ---- Extract particle masses for in-galaxy-at-z0 particles ----

    # Array to hold appropriate (z = 0 / initial) masses per particle
    # (created differently depending on settings, so initialize to dummy)
    pt_mass = None   
    ptype_z0 = hy.SplitFile(sim.get_snapshot_file(snap_z0), part_type=ptype)
    
    # Loading masses depends on particle type...
    if ptype == 1:
        # Very easy: all particles always have the same mass
        pt_mass = np.zeros(numPt, dtype=np.float32) + ptype_z0.m_dm
    else:
        # For non-DM particles, there is a difference between the initial
        # and z = 0 mass...
        if use_initial_mass:
            if ptype == 0:
                pt_mass = np.zeros(numPt, dtype=np.float32) + ptype_z0.m_baryon
            elif ptype == 4:
                snap_pt_mass = ptype_z0.InitialMass
            elif ptype == 5:
                snap_pt_mass = ptype_z0.BH_Mass
            else:
                print(f"Invalid particle type: {ptype}!")
                set_trace()
        else:
            # If we use the z = 0 particle mass, things are uniform
            snap_pt_mass = ptype_z0.Mass
            if (ptype == 0 or ptype == 4) and use_metal_mass:
                snap_pt_mass *= ptype_z0.SmoothedMetallicity

    # If the masses are not uniform, we now need the extra step of matching
    # from the snapshot masses as loaded to the particle list...
    if pt_mass is None:
        pt_snap_inds, in_snap = hy.crossref.find_id_indices(
            pt_ids, ptype_z0.ParticleIDs)
        if len(in_snap) != len(pt_ids):
            print("Why could some particles not be matched to z = 0 snap??")
            set_trace()
        pt_mass = snap_pt_mass[pt_snap_inds].astype(np.float32)

    # ----------------------------------------------------------------------

    if bin_type == 'topgal':
        output = find_top_galaxies(
            sim, pt_mass, pt_hostGal, pt_hostLograd, pt_rootGal,
            cat_gal_thissim
        )
    elif bin_type == 'origin':
        output = find_origin_matrix(
            sim, pt_mass, pt_hostGal, pt_hostLograd, pt_rootMass, pt_code,
            pt_ages, pt_rootLograd, pt_assemblyLograd, cat_gal_thissim
        )
    else:   
        print(f"Illegal bin_type '{bin_type}'!")
        set_trace()

    return output


def find_top_galaxies(
    sim, pt_mass, pt_hostGal, pt_hostLograd, pt_rootGal, cat_gal_thissim):
    """Find the N top galaxies from the pre-extracted data."""

    # Set up output array, if we find contributing galaxies
    top_mass = np.zeros((numOriginGalaxies, n_host, host_lograd_nbins))

    # Set up look-up index by host galaxy:
    lut_hostGal = SplitList(pt_hostGal, np.arange(numGal + 1))
    
    # We want to find, for each radial bin in each host, the N most
    # contributing galaxies. For this, we have to loop over both.

    # Loop through this simulation's host galaxies:
    for ihost in cat_gal_thissim:
        
        # Get the output index and particles of current host
        curr_ci = catInd[ihost]
        ind_thisHost = lut_hostGal(ihost)

        # Now, we have to find the radius bin for all particles of this host.
        ptc_host_rad = form_bin_code_radius(
            pt_hostLograd[ind_thisHost], kind='host')

        # Now loop through individual rad bins:
        lut_hostRad = SplitList(ptc_host_rad, np.arange(host_lograd_nbins + 1))
        for irad in range(host_lograd_nbins):

            # Find particles in current host/rad bin
            subind_thisRad = lut_hostRad(irad)
            if len(subind_thisRad) == 0: continue
            ind_rad = ind_thisHost[subind_thisRad]

            # Sum their masses for each individual root galaxy via histogram
            mass_by_root, edges = np.histogram(
                pt_rootGal[ind_rad], weights=pt_mass[ind_rad],
                bins=np.arange(0, max_root_gal + 2)
            )

            # Final piece: sort them and find the N most contributing galaxies
            mass_sorter = np.argsort(-mass_by_root)
            top_mass[:numOriginGalaxies, curr_ci, irad] = (
                mass_by_root[mass_sorter[:numOriginGalaxies]])

            # We put everything else into the final bin.
            top_mass[numOriginGalaxies, curr_ci, irad] = np.sum(
                mass_by_root[mass_sorter[numOriginGalaxies:]])

    return top_mass


def form_bin_code_host_ci(host_gal):
    host_ci = catInd[host_gal]
    if np.min(host_ci) < 0 or np.max(host_ci) >= n_host: 
        print("Invalid host index detected...")
        set_trace()
    return host_ci.astype(np.int8)

def form_bin_code_root_mass(root_mass):
    root_mass_span = root_mass_range[1] - root_mass_range[0]
    delta_m_root = root_mass_span / (root_mass_nbins-2)
    root_mass_code = (root_mass - root_mass_range[0]) / delta_m_root

    # We need to clip before converting to int8 to avoid under-/overflow
    root_mass_code = np.clip(root_mass_code, -127, 127).astype(np.int8)
    root_mass_code = np.clip(root_mass_code, -1, root_mass_nbins-2) + 1
    return root_mass_code

def form_bin_code_radius(pt_lograd, kind):
    if kind == 'host':
        bin_range = host_lograd_range
        num_bins = host_lograd_nbins
    elif kind == 'root':
        bin_range = root_lograd_range
        num_bins = root_lograd_nbins
    elif kind == 'ass':
        bin_range = assembly_lograd_range
        num_bins = assembly_lograd_nbins
        
    delta_r_bin = (bin_range[1] - bin_range[0]) / (num_bins - 2)
    rad_code_float = (pt_lograd - bin_range[0]) / delta_r_bin
    rad_code = np.clip(np.floor(rad_code_float), -127, 127).astype(np.int8)
    rad_code = np.clip(rad_code, -1, num_bins - 2) + 1
    return rad_code

def form_bin_code_age(pt_ages):
    """Convert float age into a discrete codes.
    Note that there are no out-of-bounds values possible here. 
    """
    delta_age = (age_range[1] - age_range[0]) / age_nbins
    age_code = np.clip((pt_ages - age_range[0]) / delta_age, -127, 127)
    return age_code.astype(np.int8)

def find_origin_matrix(
    sim, pt_mass, pt_hostGal, pt_hostLograd, pt_rootMass, pt_code, pt_ages,
    pt_rootLograd, pt_assemblyLograd, cat_gal_thissim
):
    """Bin up the pre-prepared particle data by origin code."""

    # Form bin codes
    ptc_host_ci = form_bin_code_host_ci(pt_hostGal)
    ptc_root_mass = form_bin_code_root_mass(pt_rootMass)
    ptc_host_radius = form_bin_code_radius(pt_hostLograd, kind='host')
    ptc_stellar_age = form_bin_code_age(pt_ages)
    ptc_root_radius = form_bin_code_radius(pt_rootLograd, kind='root')
    ptc_assembly_radius = form_bin_code_radius(pt_assemblyLograd, kind='ass')

    # Bin up masses with histogram
    num_pt = len(ptc_host_ci)
    full_matrix = np.zeros((num_pt, 7), dtype=np.int8)
    full_matrix[:, 0] = pt_code
    full_matrix[:, 1] = ptc_host_ci
    full_matrix[:, 2] = ptc_root_mass
    full_matrix[:, 3] = ptc_host_radius
    full_matrix[:, 4] = ptc_stellar_age
    full_matrix[:, 5] = ptc_root_radius
    full_matrix[:, 6] = ptc_assembly_radius

    bins = [
        np.arange(8), np.arange(n_host+1), np.arange(root_mass_nbins+1),
        np.arange(host_lograd_nbins+1), np.arange(age_nbins+1),
        np.arange(root_lograd_nbins+1), np.arange(assembly_lograd_nbins+1)
    ]

    set_trace()
    print("Binning up masses... ", end = '', flush = True)
    mass_binned, edg = np.histogramdd(full_matrix, weights=pt_mass, bins=bins)

    print("Sum of particle masses: ", np.sum(pt_mass))
    print("Sum of binned masses:   ", np.sum(mass_binned))

    return mass_binned


def get_particle_code(sim, root_data, particle_data):
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
    root_data : dict
        The root galaxy information for each particle.
    particle_data : dict
        The z = 0 information for each particle.

    Returns:
    --------
    code : ndarray (int8)
        Code between 0 and 6 that indicates particle origin (see above).
    """

    # Read in central galaxy information and merge list
    cantor_dir = sim.high_level_dir + '/Cantor/'
    cenGal_z0 = hd.read_data(cantor_dir + 'GalaxyTables.hdf5', 'CentralGalaxy')
    mergelist = hd.read_data(sim.spider_loc, 'MergeList')

    # Read in stellar mass at z = 0 (to distinguish qm/stripped)
    cantor_file_z0 = cantor_dir + f'/Cantor_{snap_z0:03d}.hdf5'
    mstar_z0 = hd.read_data(cantor_file_z0, '/Subhalo/MassType')[:, 4] * 1e10
    ca_shi_z0 = cantor_shi[:, snap_z0]

    root_gal = root_data['galaxies']
    host_gal = particle_data['z0_Galaxies']

    # -----------------------------------------------------
    # Decompose particles into the seven origin categories:
    # -----------------------------------------------------

    print("Decompose particles into origin categories...")

    # The simplest category is failure: no root galaxy found
    ind_tracefail = np.nonzero(root_gal < 0)[0]
    
    # Break galaxies with root into in-/ex-situ:
    ind_insitu = np.nonzero((root_gal >= 0) & (root_gal == host_gal))[0]
    ind_exsitu = np.nonzero((root_gal >= 0) & (root_gal != host_gal))[0]

    # Break ex-situ down into 'accreted' (root gal merged with z = 0 host)
    # and 'other' (everything else)
    subind_acc = np.nonzero(
        mergelist[root_gal[ind_exsitu], snap_z0] == host_gal[ind_exsitu])[0]
    subind_other = np.nonzero(
        mergelist[root_gal[ind_exsitu], snap_z0] != host_gal[ind_exsitu])[0]

    # --> Break 'ex-situ/other' down into 'root dead/alive at z = 0':
    ssubind_dead = np.nonzero(
        ca_shi_z0[root_gal[ind_exsitu[subind_other]]] < 0)[0]
    ssubind_alive = np.nonzero(
        ca_shi_z0[root_gal[ind_exsitu[subind_other]]] >= 0)[0]

    # ----> Break 'ex-situ/other/alive' down into 'in same/other FOF':
    ind_alive = ind_exsitu[subind_other[ssubind_alive]]
    sssubind_otherfof = np.nonzero(
        cenGal_z0[root_gal[ind_alive]] != cenGal_z0[host_gal[ind_alive]])[0]
    sssubind_samefof = np.nonzero(
        cenGal_z0[root_gal[ind_alive]] == cenGal_z0[host_gal[ind_alive]])[0]

    # ------> Break '.../alive/samefof' down into 'stripped/quasi-merged':
    # (If no threshold is set, we don't consider quasi-merged as an option)
    if max_mstar_qm is None:
        ssssubind_qm = np.zeros(0, dtype=int)
        ssssubind_stripped = np.arange(len(sssubind_samefof))
    else:
        ca_shi_z0_samefof = ca_shi_z0[root_gal[ind_alive[sssubind_samefof]]]
        mstar_z0_samefof = mstar_z0[ca_shi_z0_samefof]
        ssssubind_qm = np.nonzero(mstar_z0_samefof <= max_mstar_qm)[0]
        ssssubind_stripped = np.nonzero(mstar_z0_samefof > max_mstar_qm)[0]

    # ----------------- Now assign codes ----------------------------
    
    print("Assign origin codes to particles...")

    code = np.zeros(len(root_gal), dtype=np.int8) - 1

    code[ind_tracefail] = 6
    code[ind_insitu] = 0
    code[ind_exsitu[subind_acc]] = 1
    code[ind_exsitu[subind_other[ssubind_dead]]] = 5
    code[ind_alive[sssubind_otherfof]] = 4
    code[ind_alive[sssubind_samefof[ssssubind_stripped]]] = 3
    code[ind_alive[sssubind_samefof[ssssubind_qm]]] = 2
    if np.min(code < 0):
        print("Why are some galaxies not assigned a code?!?")
        set_trace()

    return code


def write_header(outloc):
    """Write header information to HDF5 file."""

    hd.write_attribute(outloc, 'Header', 'CantorCatalogue', cantorCatalogue)
    hd.write_attribute(outloc, 'Header', 'MinRatio', min_ratio)
    hd.write_attribute(outloc, 'Header', 'CompType', comp_type)
    hd.write_attribute(outloc, 'Header', 'IncludeParents', include_parents)
    hd.write_attribute(outloc, 'Header', 'ScaleRootLograd', scale_root_lograd)
    hd.write_attribute(outloc, 'Header', 'RecordAssembly', record_assembly)
    hd.write_attribute(outloc, 'Header', 'RootType', rootType)
    hd.write_attribute(outloc, 'Header', 'MaxMstarQM', max_mstar_qm)
    
    hd.write_attribute(outloc, 'Header', 'NHost', n_host)

    hd.write_attribute(outloc, 'Header', 'RootMassRange', root_mass_range)
    hd.write_attribute(outloc, 'Header', 'RootMassType', root_mass_type)
    hd.write_attribute(outloc, 'Header', 'RootMassBins', root_mass_nbins)

    hd.write_attribute(outloc, 'Header', 'HostLogradRange', host_lograd_range)
    hd.write_attribute(outloc, 'Header', 'HostLogradBins', host_lograd_nbins)

    hd.write_attribute(outloc, 'Header', 'HostRelradRange', host_relrad_range)
    hd.write_attribute(outloc, 'Header', 'HostRelradBins', host_relrad_nbins)
    
    hd.write_attribute(outloc, 'Header', 'InitialMass', use_initial_mass)
    hd.write_attribute(outloc, 'Header', 'MetalMass', use_metal_mass)



# =======================================================================
# Actual program starts here
# =======================================================================
        
def main():
    """Main program."""

    global outloc, csi_age, ptypeList_plus, n_host

    # Set up MPI to enable processing different sims in parallel
    if MPI is not None:
        comm = MPI.COMM_WORLD
        numtasks = comm.Get_size()
        rank = comm.Get_rank()
    else:
        numtasks, rank = 1, 0

    # Set up a fine interpolant to easily compute stellar ages:
    zFine = np.arange(0, 20, 0.01)
    aFine = 1 / (1 + zFine)
    ageFine = Planck13.age(zFine).value
    csi_age = interp1d(aFine, ageFine, kind='cubic', fill_value='extrapolate')

    # Load input catalogue
    cat_sim = hd.read_data(clusterCatalogue, 'Sim')
    cat_gal = hd.read_data(clusterCatalogue, 'Galaxy')
    n_host = len(cat_sim)
    cat_input = {'sim': cat_sim, 'gal': cat_gal}
    print(f"There are {n_host} galaxies in the input catalogue.")

    # Set up an extended ptype list, in case we want to find origins and 
    # trace parents
    if bin_type == 'origin' and include_parents:
        ptypeList_plus = ptypeList.copy()
        if include_parents:
            if 4 in ptypeList:
                ptypeList_plus.append(6)
            if 5 in ptypeList:
                ptypeList_plus.append(7)
    else:
        ptypeList_plus = ptypeList

    # Set up the full output matrices, since these will combine simulations.
    # They contain the mass of particles split by the different categories.
    if bin_type == 'origin':
        output_matrices = set_up_output_matrices_origin()
    else:
        output_matrices = set_up_output_matrices_topgal()

    # For simplicity, let every rank write its own output file
    outloc = outloc + '.' + str(rank) + '.hdf5'
    os.makedirs(os.path.dirname(outloc), exist_ok=True)
    if os.path.exists(outloc):
        os.rename(outloc, outloc + '.old')
    write_header(outloc)

    for isim in range(n_sim):

        # Skip this one if we're multi-threading and it's not for this task to 
        # worry about
        if not isim % numtasks == rank:
            print(f"Skipping {isim}...")
            continue

        process_sim(isim, cat_input, output_matrices)

    print("Done!")


def process_sim(isim, cat_input, output_matrices):
    """Main top-level function to process one simulation."""
    sim_stime = time.time()

    global catInd, contFlag, cantor_shi, snap_ages

    print("")
    print("**************************")
    print(f"Now processing halo CE-{isim}")
    print("**************************")
    print("")
    sys.stdout.flush()

    # Set up standard path and file names:
    if simname == "Eagle":
        sim = hy.Simulation(run_dir=basedir)
    else:
        sim = hy.Simulation(isim)
    if not os.path.exists(sim.run_dir):
        print("Can not find simulation directory...")
        return
    sim.isim = isim

    # Find which galaxies need processing from this sim
    cat_thissim = np.nonzero(cat_input['sim'] == isim)[0]
    if len(cat_thissim) == 0:
        print(f"No galaxies required from simulation {isim}, exiting...")
        return
    else:
        print(f"Simulation {isim}: {len(cat_thissim)} target galaxies.")
        cat_gal_thissim = cat_input['gal'][cat_thissim]

    # Set up files to load particles from.
    snapdir_z0 = sim.get_snapshot_file(snap_z0)
    snapdir_0 = sim.get_snapshot_file(0)
    cantor_dir = sim.high_level_dir + '/Cantor/'
    if not os.path.exists(cantor_dir):
        print(f"Cannot find cantor catalogue at '{cantorloc}'...")
        return

    # We will need the age of the Universe at z = 0 for stellar ages
    snap_ages = hy.snep_times(time_type='age', snep_list='allsnaps')
    age_now = snap_ages[snap_z0]

    # To give each run a unique identifier, use the current time
    catID = calendar.timegm(time.gmtime())
    hd.write_attribute(outloc, "Header", "CatalogueID", catID)

    # Load the full Galaxy --> Cantor SHI table (for all snapshots)
    cantor_shi = hd.read_data(cantor_dir + 'GalaxyTables.hdf5', 'SubhaloIndex')
    numGal = cantor_shi.shape[0]
    
    # Set up translation table galaxy --> index in target list 
    catInd = np.zeros(numGal, dtype=int) - 1
    catInd[cat_gal_thissim] = cat_thissim

    # Read merger and contamination info
    contFlag = hd.read_data(sim.fgt_loc, 'Full/ContFlag')[:, contFlagLevel]

    print("Done with setup, now loading particles...", flush=True)
 
    # Load particle IDs for in-galaxy-at-z=0 particles:
    particle_data = load_particles(sim)
    numPart = particle_data['NumberOfParticles']
    all_gal_z0 = particle_data['z0_Galaxies']

    if min_ratio is not None and comp_type == 'z0':
        # Set up the `comparison mass' array. This is the `ask mass'
        # that a potential host needs to have (within min_ratio) in order to 
        # be able to 'adopt' the particle. We only have to do this if we
        # compare at z = 0, otherwise it's loaded at each snapshot.                    
        if root_mass_type == 'msub':
            msub_z0 = hd.read_data(cantor_file_z0, 'Subhalo/Mass')
        elif root_mass_type == 'mstar':
            msub_z0 = hd.read_data(cantor_file_z0, 'Subhalo/MassType')[:, 4]    
        all_compMass = msub_z0[cantor_shi[all_gal_z0, snap_z0]] * 1e10

    print("Done loading particles, finding root galaxies...")

    root_data = {}
    rs_offset = snap_z0 + 1 if rootType == 'last' else 0
    root_data['snapshots'] = np.zeros(numPart, dtype = np.int8) + rs_offset
    root_data['galaxies'] = np.zeros(numPart, dtype = np.int32) - 1
    root_data['masses'] = np.zeros(numPart, dtype = np.float32) - 1
    root_data['logradii'] = np.zeros(numPart, dtype = np.float32) + 1000
    if record_assembly:
        root_data['assembly_times'] = np.zeros(numPart, dtype=np.float32) - 1
        root_data['assembly_radii'] = np.zeros(numPart, dtype=np.float32) - 1
    else:
        root_data['assembly_times'] = None
        root_data['assembly_radii'] = None

    # === Core processing step: find root galaxies for all particles ===
    for isnap in range(5): # nsnap):
        root_data = process_snapshot(
            isnap, sim, root_data, particle_data, cantor_shi)

    print("")
    print("Now determining origin code for all particles...")
    
    # Determine 'type-code' for all (in-z0-gal) particles:
    all_code = get_particle_code(sim, root_data, particle_data)

    print("Binning up particle masses for each type...")    
    for ptype in ptypeList:

        # *** Different in this version: instead of origin code, we find
        # *** the 30 most-contributing galaxies in each radial bin (for each
        # *** host separately). All others are grouped in `gal 30'.
        mass_binned = bin_particles(
            sim, particle_data, root_data, all_code, ptype, cat_gal_thissim)

        # Now combine results across simulations and write
        output_matrices[ptype][isim, ...] = mass_binned.astype(np.float32)
        ptname = ptype_names[ptype]
        hd.write_data(outloc, f'BinnedMasses_{ptname}', output_matrices[ptype])


def set_up_output_matrices_topgal():
    """Set up the full output matrices."""
    data = {}
    for ptype in [0, 1, 4, 5]:
        if ptype in ptypeList:
            mass_binned = np.zeros(
                (n_sim, numOriginGalaxies, n_host, host_lograd_nbins),
                dtype=np.float32
            )
            data[ptype] = mass_binned
    return data


def set_up_output_matrices_origin():
    """Set up the full output matrices."""
    data = {}
    for ptype in [0, 1, 4, 5, 6, 7]:
        if ptype in ptypeList_plus:
            mass_binned = np.zeros(
                (n_sim, 7, n_host, root_mass_nbins, host_lograd_nbins,
                 age_nbins, root_lograd_nbins, assembly_lograd_nbins),
                dtype=np.float32
            )
            data[ptype] = mass_binned
    return data



if __name__ == "__main__":
    main()
    print("Done!")
