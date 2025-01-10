import sys
import numpy as np
from ase import Atoms, units
from ase.io import read, iread, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammps import convert

# Code based on the LAMMPS IO package from ASE
# However, this code reads and writes configurations one at a time from a LAMMPS trajectory file
# as opposed to using a deque and loading all configurations into memory at once like ASE does.
# This is more suitable for larger trajectory files that will not fit onto RAM.


def main():
    infile = sys.argv[1]
    trajname = sys.argv[2]
    lammps_dump_text_to_traj(infile, trajname, units=sys.argv[3])


def lammps_data_to_ase_atoms(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    """
    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int)

        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    # !TODO: how need quaternions be converted?
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions is not None:
        out_atoms = Quaternions(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            quaternions=quaternions,
        )
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges(charges)
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(out_atoms, energy=0.0,
                                           forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute or fix (but not the quaternian)
        if (colname.startswith('f_') or colname.startswith('v_') or
                (colname.startswith('c_') and not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='float')

    return out_atoms


def construct_cell(diagdisp, offdiag):
    """Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def lammps_dump_text_to_traj(infile, trajname, **kwargs):
    """Process cleartext lammps dumpfiles

    :param fileobj: filestream providing the trajectory data
    :param index: integer or slice object (default: get the last timestep)
    :returns: list of Atoms objects
    :rtype: list
    """
    # Load all dumped timesteps into memory simultaneously
    #lines = deque(fileobj.readlines())
    #index_end = get_max_index(index)

    n_atoms = 0
    images = []

    traj = Trajectory(trajname, "w")

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc = None, None, False

    with open(infile, "r") as fh:
        while True:
            line = fh.readline()

            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                n_atoms = 0
                line = fh.readline()
                print("TIMESTEP", line)

            if "ITEM: NUMBER OF ATOMS" in line:
                line = fh.readline()
                n_atoms = int(line.split()[0])

            if "ITEM: BOX BOUNDS" in line:
                tilt_items = line.split()[3:]
                celldatarows = [fh.readline() for _ in range(3)]
                celldata = np.loadtxt(celldatarows)
                diagdisp = celldata[:, :2].reshape(6, 1).flatten()

                # determine cell tilt (triclinic case!)
                if len(celldata[0]) > 2:
                    # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
                    # to assign tilt (vector) elements ...
                    offdiag = celldata[:, 2]
                    # ... otherwise assume default order in 3rd column
                    # (if the latter was present)
                    if len(tilt_items) >= 3:
                        sort_index = [tilt_items.index(i)
                                      for i in ["xy", "xz", "yz"]]
                        offdiag = offdiag[sort_index]
                else:
                    offdiag = (0.0,) * 3

                cell, celldisp = construct_cell(diagdisp, offdiag)

                # Handle pbc conditions
                if len(tilt_items) == 3:
                    pbc_items = tilt_items
                elif len(tilt_items) > 3:
                    pbc_items = tilt_items[3:6]
                else:
                    pbc_items = ["f", "f", "f"]
                pbc = ["p" in d.lower() for d in pbc_items]

            if "ITEM: ATOMS" in line:
                colnames = line.split()[2:]
                datarows = [fh.readline() for _ in range(n_atoms)]
                data = np.loadtxt(datarows, dtype=str)
                out_atoms = lammps_data_to_ase_atoms(
                    data=data,
                    colnames=colnames,
                    cell=cell,
                    celldisp=celldisp,
                    atomsobj=Atoms,
                    pbc=pbc,
                    **kwargs
                )
                traj.write(out_atoms)


if __name__ == "__main__":
    main()
