#!/bin/bash


startframe=0
endframe=5001
logfreq=100

trajectory=../lammps_example/prod_control.traj
reference_mol=../lammps_example/spc.xyz
atom_indices=../lammps_example/all_atom_indices.npy

timestep=0.004 
sigma=2
temperature=298.0
eMD=0.0

python decompose_velocities.py $trajectory $logfreq $atom_indices $startframe  $endframe
python two_phase_dos.py $trajectory $timestep $sigma $reference_mol $temperature $eMD
python plot_2pt.py
