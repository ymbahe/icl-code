# icl-code
Code for ICL origin paper

This repository contains the following python scripts:

- `build_cluster_catalogue.py`: assembles a catalogue of clusters from the simulation outputs. Apart from "basic" properties such as M200, R200, stellar mass, it also computes the time of the last major merger (using an adjustable M200 mass threshold), which is stored as `RelaxTime`.

- `cantor_example_particles.py`: extract particles identified by Cantor as belonging to one particular galaxy, including those particles that are also found by Subfind and those that are in the same galaxy in previous snapshots. For visualisation purposes only.

- `collect_profiles.py`: combine profile outputs for individual clusters into a single file.

- `combine_origin_tables.py`: combine "origin tables" produced by individual MPI ranks into a single file (used to post-process outputs from XXX).

- `find_clusters_z0p25.py`: find clusters above a given mass threshold in a given snapshot in each simulation, and assemble a script to generate images of all of them (currently `cluster_image.py`, but this is simply an earlier version of the multi-purpose `galaxy_image.py` script).

- `find_origin_contributions.py`: modified version of `find_particle_root_galaxy_cantor_xd.py` (see below) that finds the N most contributing galaxies instead.

- `find_particle_root_galaxy_cantor_xd.py`: script to extract the "root" galaxy of each particle in each galaxy in an input catalogue (typically the centrals of clusters, as found with `build_cluster_catalogue.py`). This is either the galaxy in which the particle was formed or in which it was last before its z = 0 galaxy. Optionally, the same information is found for the gas parent particles of star/BH particles (i.e. what was the first/last galaxy with which the progenitor gas particle was associated before turning into a star/BH). The results are then combined into an "origin matrix", a 9-dimensional array that records the total mass of particles per (0) simulation, (1) origin code, (2) z = 0 host, (3) -dummy-, (4) root mass, (5) host radius, (6) -dummy-, (7) age, (8) root radius.

  The 7 origin codes are:

  | Value | Meaning | Comment | 
  |---|---|---|
  | 0 | in-situ | root galaxy == host | 
  | 1 | accreted  | root galaxy merged with host |
  | 2 | quasi-merged | root galaxy still alive, but with stellar mass below a certain threshold |
  | 3 | stripped | root galaxy is still alive and its stellar mass is above the threshold | 
  | 4 | stolen | root galaxy is alive but in a totally different FOF halo than the host | 
  | 5 | adopted | root galaxy is not alive anymore, but it did not merge with the host | 
  | 6 | tracefail | no root galaxy could be determined | 

## To do items

- Compare `find_particle_root_galaxy_cantor_xd.py` and `find_origin_contributions.py` to see what differences there are. It would be ideal to merge both into one file since they share very substantial code.
- Check through `find_particle_root_galaxy_cantor_xd.py` to verify correct working, add (more) comments
- Add host radius in first in-host snapshot as an additional dimension of output (can use currently empty indices 3 or 6).
