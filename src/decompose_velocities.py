import sys
import time
import json
import numpy as np
from scipy import sparse
from numba import njit
from ase.io.trajectory import Trajectory
from ase.neighborlist import get_connectivity_matrix, natural_cutoffs, NeighborList


logfreq = int(sys.argv[2])
indices = np.load(sys.argv[3])
start = int(sys.argv[4])
stop = int(sys.argv[5])
traj = Trajectory(sys.argv[1])[start:stop]

# Assumptions:
# 1) molecular identity does not change over the simulation
# 2) the trajectory has been preprocesed to only include molecules you wish to calculate over
#    the simulation length you wish to calculate
# 3) Number of molecules stays constant

@njit
def center_of_mass(masses, positions):
    numerator = np.dot(masses, positions)
    denominator = np.sum(masses)
    r_com = numerator / denominator
    return r_com

@njit
def velocity_center_of_mass(masses, velocities):
    numerator = np.dot(masses, velocities)
    denominator = np.sum(masses)
    v_com = numerator / denominator
    return v_com


@njit
def inertia_tensor(masses, positions, relative = False):
    if relative:
        r = positions
    else:
        r_com = center_of_mass(masses, positions)
        r = positions - r_com

    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(masses)):
        x, y, z = r[i]
        m = masses[i]
        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])
    return Itensor

@njit
def principle_moment_of_intertia(Itensor):
    evals, evecs = np.linalg.eigh(Itensor)
    return evals

@njit
def total_angular_momentum(masses, positions, velocities, relative = False):
    L = np.sum(angular_momentum(masses, positions, velocities, relative), axis=0)
    return L

@njit
def angular_momentum(masses, positions, velocities, relative = False):
    if relative:
        r = positions
        v = velocities
    else:
        r_com = center_of_mass(masses, positions)
        r = positions - r_com
        v_com = velocity_center_of_mass(masses, velocities)
        v = velocities - v_com

    momenta = masses[:, np.newaxis] * v
    L = np.cross(r, momenta)
    return L

@njit
def angular_velocity(masses, positions, velocities, principle = False, relative = False):
    L = total_angular_momentum(masses, positions, velocities, relative) # amu * angstrom**2 / time_ase
    Itensor = inertia_tensor(masses, positions, True) # amu * angstrom**2

    if principle:
        principle_moments = principle_moment_of_intertia(Itensor)
        Itensor = np.array(
                [[principle_moments[0],0., 0.],
                [0., principle_moments[1], 0.],
                [0., 0., principle_moments[2]]
                ])

    else:
        pass

    Itensor_inv = np.linalg.inv(Itensor)
    omega = (Itensor_inv @ L.T).T
    return omega







@njit
def rotational_velocity(masses, positions, velocities, principle=False, relative = False):
    omega = angular_velocity(masses, positions, velocities, principle, relative)
    if relative:
        r = positions
    else:
        r_com = center_of_mass(masses, positions)
        r = positions - r_com

    v_rot = np.cross(omega,r)
    return v_rot


@njit
def rotational_velocity_new(m_mol, rel_r_mol, rel_v_mol):
    L = total_angular_momentum(m_mol, rel_r_mol, rel_v_mol, relative = True)
    Itensor = inertia_tensor(m_mol, rel_r_mol, relative = True)

    # since Itensor is symettric, use eigh to be faster and sort the eigenvectors
    # eval[i] is the eigenvalue with eigenvector evecs[:,i] (by columns)
    evals, evecs = np.linalg.eigh(Itensor)

    # if linear, evals[2] = 0

    # Angular Velocity along principle axis
    p_omega = np.zeros(3)
    for i in range(3):
        for k in range(3):
            if evals[i] > 0:
                p_omega[i] += L[k] * evecs[k][i]/evals[i]

    # Inertia Weighted angular velocity
    omega = np.zeros(3)
    angularv = np.zeros(3)
    for i in range(3):
        for k in range(3):
            if evals[i] > 0:
                # angular velocity
                omega[i] += p_omega[k] * evecs[i][k]
                # angular velocity weighted by principle moments of inertia
                angularv[i] += p_omega[k] * evecs[i][k] * np.sqrt(evals[k])

    # Rotational velocity for the atoms
    vrot = np.cross(omega, rel_r_mol)
    return vrot

@njit
def angular_velocity_new(m_mol, rel_r_mol, rel_v_mol):
    L = total_angular_momentum(m_mol, rel_r_mol, rel_v_mol, relative = True)
    Itensor = inertia_tensor(m_mol, rel_r_mol, relative = True)

    # since Itensor is symettric, use eigh to be faster and sort the eigenvectors
    # eval[i] is the eigenvalue with eigenvector evecs[:,i] (by columns)
    evals, evecs = np.linalg.eigh(Itensor)

    # if linear, evals[2] = 0

    # Angular Velocity along principle axis
    p_omega = np.zeros(3)
    for i in range(3):
        for k in range(3):
            if evals[i] > 0:
                p_omega[i] += L[k] * evecs[k][i]/evals[i]

    # Inertia Weighted angular velocity
    omega = np.zeros(3)
    angularv = np.zeros(3)
    for i in range(3):
        for k in range(3):
            if evals[i] > 0:
                # angular velocity
                omega[i] += p_omega[k] * evecs[i][k]
                # angular velocity weighted by principle moments of inertia
                angularv[i] += p_omega[k] * evecs[i][k] * np.sqrt(evals[k])

    ## Rotational velocity for the atoms
    #vrot = np.cross(omega, rel_r_mol)
    return angularv


frame = 0
nframes = len(traj)

#temps = np.zeros(nframes)
#energies = np.zeros(nframes)

for atoms in traj:

    if frame == 0:
        cutoffs = natural_cutoffs(atoms, mult = 0.85)
        neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighbor_list.update(atoms)
        matrix = neighbor_list.get_connectivity_matrix()

        nmolecules, component_list = sparse.csgraph.connected_components(matrix)
        print("No. Molecules:", nmolecules)

        mol_idx_dict = {}
        for mol_idx in range(nmolecules):
            atom_idxs = [i for i in range(len(component_list)) if component_list[i] == mol_idx]
            mol_idx_dict[mol_idx] = atom_idxs

        np.save('component_list.npy', component_list)
        with open("mol_idxs.json", "w") as f:
            json.dump(mol_idx_dict, f)

        m = atoms.get_masses()
        np.save('masses.npy', m)

        natoms = len(atoms)

        vtot = np.zeros(shape=(nframes, natoms, 3))
        vtrn = np.zeros(shape=(nframes, natoms, 3))
        vrot = np.zeros(shape=(nframes, natoms, 3))
        omegarot = np.zeros(shape=(nframes, nmolecules, 3))
        vvib = np.zeros(shape=(nframes, natoms, 3))

    else:
        pass

    if frame % logfreq == 0: print(f"Processing frame {frame+1}/{nframes}")

    r = atoms.get_positions()
    v = atoms.get_velocities()
    #temps[frame] = atoms.get_temperature()
    #energies[frame] = atoms.get_potential_energy()

    # Loop over molecules
    for mol_idx in range(nmolecules):
        atom_idxs = mol_idx_dict[mol_idx]

        r_mol = r[atom_idxs]
        m_mol = m[atom_idxs]
        v_mol = v[atom_idxs]

        # Find COM position and COM velocity
        v_com = velocity_center_of_mass(m_mol, v_mol)
        r_com = center_of_mass(m_mol, r_mol)

        # Set the COM to the origin
        rel_r_mol = r_mol - r_com
        rel_v_mol = v_mol - v_com

        vrot_mol = rotational_velocity_new(m_mol, rel_r_mol, rel_v_mol)
        omegarot[frame][mol_idx] = angular_velocity_new(m_mol, rel_r_mol, rel_v_mol)

        #vrot_mol = rotational_velocity(m_mol, r_mol, v_mol, principle=False)
        #omegarot[frame][mol_idx] = angular_velocity(m_mol, r_mol, v_mol, principle = True)

        #Itensor_mol = inertia_tensor(m_mol, r_mol)
        #principle_mi[frame][mol_idx] = principle_moment_of_intertia(Itensor_mol)

        for j, atom_idx in enumerate(atom_idxs):
            vtrn_atom = v_com
            vrot_atom = vrot_mol[j]
            vvib_atom = rel_v_mol[j] - vrot_atom
            vtot_atom = v[atom_idx]

            vtrn[frame][atom_idx] = vtrn_atom
            vrot[frame][atom_idx] = vrot_atom
            vvib[frame][atom_idx] = vvib_atom
            vtot[frame][atom_idx] = vtot_atom

    frame += 1

np.save("vtrn.npy", vtrn)
np.save("vrot.npy", vrot)
np.save("vvib.npy", vvib)
np.save("vtot.npy", vtot)

np.save("vangular.npy", omegarot)

#np.save("principle_moments_of_inertia.npy", principle_mi)

#np.save("energy.npy", energies)
#np.save("temperature.npy", temps)
