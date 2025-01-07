import sys
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
from scipy import constants
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from scipy.optimize import fsolve


traj = Trajectory(sys.argv[1])
timestep = float(sys.argv[2]) # ps
sigma = int(sys.argv[3]) # symmetry number
reference_molecule = read(sys.argv[4])

T = float(sys.argv[5]) # K
eMD = float(sys.argv[6]) # eV


nframes = len(traj)
init_config = traj[0]

volume = init_config.get_volume() # angstrom**3
masses = init_config.get_masses() # g/mol
natoms = len(init_config)
beta = 1.0/(T * units.kB) # 1/eV

reference_mass = np.sum(reference_molecule.get_masses())
nmolecules = natoms // len(reference_molecule)

pmi = reference_molecule.get_moments_of_inertia() #  amu*angstrom**2
pmi_si  = pmi / units.mol / 1000.0 / 1.0E20 # kg * m^2

def calculate_massweighted_vdos(v, beta, masses, timestep, factor = 2.0):
    # Assume beta is in 1/eV, m is in amus, the timestep is in ps, and the velocities are in angstrom / ASE time units
    c_cmpers = constants.c * 100.0
    ps_to_s = 1E-12

    natoms = len(v[0])
    nframes = len(v)

    ft_v = np.fft.rfft(v, axis=0)  # Each atom, each xyz-component, time series is the frame idx
    ft_v_sq = (ft_v * np.conjugate(ft_v))
    sv = np.real(ft_v_sq) / nframes
    ft_mvdos_tot = np.zeros(len(sv))

    for j in range(natoms):
        mj = masses[j]
        for k in range(3):
            ft_mvdos_tot += mj * sv[:,j,k]

    ft_mvdos_tot *= factor*beta # units of inverse frequency
    ft_mvdos_tot_cm = ft_mvdos_tot * timestep * ps_to_s * c_cmpers # units of cm

    freq = np.fft.rfftfreq(nframes, d=timestep) # units of frequency (1/ps)
    freq_wavenumber = freq  / ps_to_s / c_cmpers # units of inverse cm

    return freq_wavenumber, ft_mvdos_tot_cm


def diffusion_coefficient(beta, m, N, Sv):
    s0 = Sv[0].real  # s0 = 12*m*N*D/kB/T units of cm (inverse frequency)
    c_cmpers = constants.c * 100.0
    s0 = s0 / c_cmpers
    m = m / units.mol / 1000.0 # units of kg
    beta *= units.J # 1/Joules
    rhs = 12.0 * m * N * beta
    return s0 / rhs  # m^2/s

def dimensionless_diffusivity(beta, V, N, m, Sv):

    s0 = Sv[0].real  # s0 = 12*m*N*D/kB/T units of cm (inverse frequency)
    c_cmpers = constants.c * 100.0
    s0 /= c_cmpers # units of seconds
    m = m / units.mol / 1000.0 # units of kg
    V = V * 1.0E-30 # units of m^3
    beta = beta * units.J # 1/Joules

    A = 2.0*s0/9.0/N
    B = np.pi / beta / m
    C = N/V
    D = 6.0/np.pi

    delta = A * (B**0.5) * (C**(1.0/3.0)) * (D**(2.0/3.0))
    return delta

def fluidicity(f, diff):
    A = 2.0 *(f**7.5) / (diff**4.5)
    B = 6.0 *(f**5) / (diff**3)
    C = (f**3.5) / (diff**1.5)
    D = 6.0 *(f**2.5) / (diff**1.5)
    E = 2.0 * f
    F = 2.0
    return A - B - C + D + E - F

def derivative_fluidicity(f, diff):
    A = 2.0 *(7.5*f**6.5) / (diff**4.5)
    B = 6.0 *(5*f**4) / (diff**3)
    C = (3.5*f**2.5) / (diff**1.5)
    D = 6.0 *(2.5*f**1.5) / (diff**1.5)
    E = 2.0
    return A - B - C + D + E

def get_fluidicity(diff, f = 0.5, niter=100):
    converged = False

    print("Solving for fluidicity")
    print("Iter Fluidicity Convergence")

    for count in range(niter):
        fn = f - fluidicity(f, diff) / derivative_fluidicity(f, diff)
        if fn < 0.0:
            fn = 0.0
        if np.abs(fn - f) <= 1E-15:
            converged = True
        f = fn
        print(count, fn, fluidicity(fn, diff))
        if converged:
            break
    print()
    return fn

def Sv_2pt(Sv, freq, f, N):
    s0 = Sv[0]
    denominator = 1.0 + (np.pi *s0*freq/6.0/f/N)**2
    Sv_gas = s0/denominator
    Sv_solid = Sv - Sv_gas
    return {"gas": Sv_gas, "solid": Sv_solid}

def calculate_and_smooth_powerspectra(vfile, beta, masses, timestep, factor=2.0, smooth = False):
    print("Calculating power spectra with", vfile,"...")
    freq, Sv = calculate_massweighted_vdos(np.load(vfile), beta, masses, timestep, factor)
    if smooth:
        Sv = savgol_filter(Sv, 100, 3)
    else:
        pass
    return freq, Sv


#### THERMO PROPERTIES
def reference_energy(eMD, ndof, ftrn, frot, beta):
    '''
    eMD is the average total (kinetic + potential) energy from an MD simulation
    to ensure that the reference is the sum of harmonic oscilators, we subtract out the kinetic energy
    in terms of fluidicity and the NDOF obtained from the total power spectra
    '''
    #e0 = eMD - 3.0*N*(1.0-0.5*ftrn-0.5*frot)/beta
    e0 = eMD - ndof*(1.0-0.5*ftrn-0.5*frot)/beta
    return e0

def solid_weights(beta, freq):
    """
    solid weights are used for trn, rot, and vib modes
    beta is in units of 1/eV, freq is in wavenumber (1/cm)
    """
    h = constants.h # Joule - seconds
    freq_seconds = freq * constants.c * 100.0 # 1/seconds
    beta *= units.J # 1/Joules

    arg = h*beta*freq_seconds

    w_E = 0.5*arg + arg / (np.exp(arg) - 1.0)
    w_S = arg/(np.exp(arg)-1.0) - np.log(1.0-np.exp(-arg))

    w_A = w_E - w_S
    #w_A = np.log((1.0 - np.exp(-arg)/(np.exp(-0.5*arg))))

    print("Solid weights")
    print(w_E[0:5])
    print(w_S[0:5])
    print(w_A[0:5])
    print()
    weights = {
            "energy": w_E,
            "entropy": w_S,
            "helmholtz": w_A,
            }

    return weights


def gas_trn_weights(ftrn, difftrn, N, V, T, m):
    w_E = 0.5
    w_S = hard_sphere_entropy_over_k(ftrn, difftrn, N, V, T, m)/3.0
    w_A = w_E - w_S

    weights = {
            "energy": w_E,
            "entropy": w_S,
            "helmholtz": w_A,
            }

    return weights


def hard_sphere_entropy_over_k(ftrn, difftrn, N, V, T, m):
    V = V * 1.0E-30 # units of m^3
    m = m / units.mol / 1000.0 # units of kg

    y = ftrn**2.5 / difftrn**1.5
    z = carnahan_starling(y)

    arg1 = 2.0*np.pi*m*constants.k*T/constants.h/constants.h
    arg2 = V * z / ftrn / N

    term1 = np.log((arg1**1.5) * arg2)

    numerator = y*(3.0*y-4.0)
    denominator = (1.0-y)**2
    term2 = numerator / denominator

    return 2.5 + term1 + term2


def carnahan_starling(y):
    numerator = 1.0 + y + y*y - y*y*y
    denominator = (1.0-y)**3
    return numerator / denominator


def gas_rot_weights(sigma, T, pmi):
    w_E = 0.5
    w_S = rigid_rotation_entropy_over_k(sigma, T, pmi)/3.0
    w_A = w_E - w_S

    weights = {
            "energy": w_E,
            "entropy": w_S,
            "helmholtz": w_A,
            }

    return weights

def rigid_rotation_entropy_over_k(sigma, T, pmi):
    term1 = np.sqrt(np.pi)*np.exp(1.5) / sigma
    A = T / rotational_temperature(pmi[0])
    B = T / rotational_temperature(pmi[1])
    C = T / rotational_temperature(pmi[2])
    term2 = np.sqrt(A*B*C)

    return np.log(term1*term2)

def rotational_temperature(pmi):
    numerator = constants.h * constants.h
    denominator = 8.0 * np.pi *np.pi * pmi * constants.k
    return numerator / denominator


def two_phase_integral(freq, mode, Sv, w_solid, w_gas = None):

    print(mode)
    integrand = Sv['solid']*w_solid[mode]
    integrand[0] = 0.0 # Hack to account for 0 frequency solid modes are 0 for the DOS and infinite in the weight formulas
    print("Solid Dos:", Sv["solid"])
    print("Solid Weight:", w_solid[mode])
    print("Solid Integrand:", integrand)
    solid = simpson(integrand, x = freq)

    if w_gas is not None:
        integrand = Sv['gas']*w_gas[mode]
        print("Gas Dos:", Sv["gas"])
        print("Gas Weight:", w_gas[mode])
        print("Gas Integrand:", integrand)
        gas = simpson(integrand, x = freq)
        m = gas + solid
    else:
        m = solid

    print()
    return m


### RUN THE ANALYISIS
freq_tot, Sv_tot = calculate_and_smooth_powerspectra('vtot.npy', beta, masses, timestep)
freq_trn, Sv_trn = calculate_and_smooth_powerspectra('vtrn.npy', beta, masses, timestep)
freq_rot, Sv_rot = calculate_and_smooth_powerspectra('vangular.npy', beta, np.ones(len(masses)), timestep)
freq_vib, Sv_vib = calculate_and_smooth_powerspectra('vvib.npy', beta, masses, timestep)

ndof = simpson(Sv_tot, x=freq_tot)
print("Integral of total DOS (total degrees of freedom):", ndof)
print("3*N_atoms:", 3*natoms)

print("D tot:", diffusion_coefficient(beta, reference_mass, nmolecules, Sv_tot))
print("D trn:", diffusion_coefficient(beta, reference_mass, nmolecules, Sv_trn))
print("D rot:", diffusion_coefficient(beta, reference_mass, nmolecules, Sv_rot))
print("D vib:", diffusion_coefficient(beta, reference_mass, nmolecules, Sv_vib))

diff_trn = dimensionless_diffusivity(beta, volume, nmolecules, reference_mass, Sv_trn)
diff_rot = dimensionless_diffusivity(beta, volume, nmolecules, reference_mass, Sv_rot)

print("Translation fluidicity")
f_trn = get_fluidicity(diff_trn, f=1.0)
print("Rotation fluidicity")
f_rot = get_fluidicity(diff_rot, f=0.1)

Sv_trn_2pt = Sv_2pt(Sv_trn, freq_trn, f_trn, nmolecules)
Sv_rot_2pt = Sv_2pt(Sv_rot, freq_rot, f_rot, nmolecules)
# All vibrational modes are treated as harmonic, therefore the gas contribution for vibrations is 0
Sv_vib_2pt = {"gas": np.zeros(len(Sv_vib)), "solid": Sv_vib}

print("Translation Diffusion:", diff_trn)
print("Translation Fluidicity:", f_trn)
print("Rotational Diffusion:", diff_rot)
print("Rotation Fluidicity:", f_rot)

w_solid_trn = solid_weights(beta, freq_trn)
w_solid_rot = solid_weights(beta, freq_rot)
w_solid_vib = solid_weights(beta, freq_vib)

w_gas_trn = gas_trn_weights(f_trn, diff_trn, nmolecules, volume, T, reference_mass)
w_gas_rot = gas_rot_weights(sigma, T, pmi_si)

e0 = reference_energy(eMD, ndof, f_trn, f_rot, beta)

energy_components = {
        "energy": {
            "reference": e0,
            "trn": two_phase_integral(freq_trn, 'energy', Sv_trn_2pt, w_solid_trn, w_gas_trn) / beta / nmolecules,
            "rot": two_phase_integral(freq_rot, 'energy', Sv_rot_2pt, w_solid_rot, w_gas_rot) / beta,
            "vib": two_phase_integral(freq_vib, 'energy',  Sv_vib_2pt, w_solid_vib, None) / beta,
            },
        "entropy": {
            "trn": two_phase_integral(freq_trn, 'entropy', Sv_trn_2pt, w_solid_trn, w_gas_trn) * units.kB / nmolecules,
            "rot": two_phase_integral(freq_rot, 'entropy', Sv_rot_2pt, w_solid_rot, w_gas_rot) * units.kB / nmolecules,
            "vib": two_phase_integral(freq_vib, 'entropy',  Sv_vib_2pt, w_solid_vib, None) * units.kB / nmolecules,
            },
        "helmholtz": {
            "reference": e0,
            "trn": two_phase_integral(freq_trn, 'helmholtz', Sv_trn_2pt, w_solid_trn, w_gas_trn) / beta / nmolecules,
            "rot": two_phase_integral(freq_rot, 'helmholtz', Sv_rot_2pt, w_solid_rot, w_gas_rot) / beta / nmolecules,
            "vib": two_phase_integral(freq_vib, 'helmholtz',  Sv_vib_2pt, w_solid_vib, None) / beta / nmolecules,
            },
}

print(energy_components)
for energy, value_dict in energy_components.items():
    total = 0
    for mode, value in value_dict.items():
        total += value
    print(energy, total * units.mol/units.J)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(3.5,6.5), dpi=600, layout='constrained')

ax = axs[0]
ax.plot(freq_tot, Sv_tot, label = "Total")
ax.plot(freq_trn, Sv_trn, label = "Trn")
ax.plot(freq_rot, Sv_rot, label = "Rot")
ax.plot(freq_vib, Sv_vib, label = "Vib")

ax.set_xlim((0, 1000))
#ax.set_ylim((0, 12))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")
ax.legend()

ax = axs[1]
ax.plot(freq_trn, Sv_trn, label = "Trn")
ax.plot(freq_trn, Sv_trn_2pt['gas'], label = "Trn(g)")
ax.plot(freq_trn, Sv_trn_2pt['solid'], label = "Trn(s)")

ax.set_xlim((0, 500))
#ax.set_ylim((0, 12))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")

ax.legend()

ax = axs[2]
ax.plot(freq_rot, Sv_rot, label = "Rot")
ax.plot(freq_rot, Sv_rot_2pt['gas'], label = "Rot(g)")
ax.plot(freq_rot, Sv_rot_2pt['solid'], label = "Rot(s)")

ax.set_xlim((0, 1000))
#ax.set_ylim((0, 4))
ax.set_xlabel(r"$\nu$ (1/cm)")
ax.set_ylabel(r"S($\nu$) (cm)")

ax.legend()

fig.savefig('Sv.png')
