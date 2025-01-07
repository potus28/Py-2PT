import numpy as np
natoms = 1536 # 512 * 3
x = np.arange(natoms, dtype = int)
np.save("all_atom_indicies.npy", x)
