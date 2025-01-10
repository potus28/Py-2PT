#!/bin/bash

lmp -in lmp.in
python get_all_atom_indicies.py
python lammps_to_traj.py prod_control.lammpstrj prod_control.traj real
