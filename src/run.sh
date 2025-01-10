


startframe=0
endframe=5001

timestep=0.004 
logfreq=100
sigma=2
temperature=298.0
eMD=0.0

python decompose_velocities.py ../lammps_example/prod_control.traj $logfreq ../../all_atom_indicies.npy $startframe  $endframe
python two_phase_dos.py ../lammps_example/prod_control.traj $timestep $sigma ../../spc.xyz $temperature $eMD


