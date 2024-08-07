# %% [markdown]
# This script runs HMC including Teperature Gradients & weak lines for gamma self
# 2024/08/06 Updated:Using the Transmittance calculation along wavenumber, lpf.xsmatrix, print/plot refactoring

##Read (& Show) the spectra data
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from HMC_defs_4data import (
    print_spectra_data,
    plot_spectra_data,
    create_mdbs,
    create_opapremodit,
    print_parameters,
    plot_save_results,
)

# Road the spectra file
direc1 = "../../実験/Data/240701/Corrected/"
direc2 = "../../実験/Data/240701/Corrected/"
direc3 = "../../実験/Data/240701/Corrected/"
direc4 = "../../実験/Data/240703/Corrected/"

file1 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_300K_wmask_1103_1_AveNorm_fringe_removed.dat"
file2 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_500K_wmask_45dMt-X0V-Y75V_1322_1_AveNorm_fringe_removed.dat"
file3 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_700K_wmask_45dMt-X75V-Y0V_1515_1_AveNorm_fringe_removed.dat"
file4 = "240703_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_1000K_wmask_45dMt-150V-Y150V_1416_1_AveNorm_fringe_removed.dat"

Data1 = np.genfromtxt(direc1 + file1, delimiter=" ", skip_header=0)
Data2 = np.genfromtxt(direc2 + file2, delimiter=" ", skip_header=0)
Data3 = np.genfromtxt(direc3 + file3, delimiter=" ", skip_header=0)
Data4 = np.genfromtxt(direc4 + file4, delimiter=" ", skip_header=0)

# Trim the data amount
start = 8712  # λ=1628.235:11294, 1617.32:6928, 1621.78:8712
end = 8773
wavd1 = Data1[start:end, 0]
Trans1 = Data1[start:end, 1]
wavd2 = Data2[start:end, 0]
Trans2 = Data2[start:end, 1]
wavd3 = Data3[start:end, 0]
Trans3 = Data3[start:end, 1]
wavd4 = Data4[start:end, 0]
Trans4 = Data4[start:end, 1]

print_spectra_data(wavd1, Trans1, wavd2, Trans2, wavd3, wavd4)
plot_spectra_data(wavd1, Trans1, wavd2, Trans2, wavd3, Trans3, wavd4, Trans4)
plt.close()  # close the plot window


# Import the modules for Running HMC
from exojax.utils.constants import Tref_original, Tc_water  # [bar/atm]
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbHitemp
from exojax.spec.hitran import line_strength
from exojax.spec.specop import SopInstProfile
from Isobaric_Numdensity import calc_dnumber_isobaric
from Trans_model_1Voigt_HITEMP_nu_2408rev import Trans_model_1Voigt
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random


# parameter Setting
Tarr1 = (
    jnp.array([23.5, 23.6, 23.4, 23.5, 23.4, 23.4, 23.4, 23.3]) + Tc_water
)  # CH3-10 of 2024/07/01 at 300K
Tarr2 = (
    jnp.array([228.8, 227.7, 230.3, 231.5, 231.0, 231.7, 234.2, 219.6]) + Tc_water
)  # CH3-10 of 2024/07/01 at 500K
Tarr3 = (
    jnp.array([426.3, 426.4, 431.9, 433.3, 431.7, 431.8, 434.7, 413.5]) + Tc_water
)  # CH3-10 of 2024/07/01 at 700K
Tarr4 = (
    jnp.array([720.5, 720.0, 727.2, 734.3, 733.8, 731.4, 729.7, 725.2]) + Tc_water
)  # CH3-10 of 2024/07/03 at 1000K

T_seal = 23.6 + Tc_water  # [K], The temperature at the time the cell was sealed
Twt = 1000  # Weighting Temperature
Resolution = 0.0025  # Data grid resolutionin wavelength[nm]
VMR = 1  # volume mixing ratio
L = 49.7  # path length[cm]
P0_total_1 = 0.423  # Pressure at the time the cell was sealed
P0_total_2 = 0.423
P0_total_3 = 0.423
P0_total_4 = 0.423

# calculate the number density and pressure at given temperatures
ngas1, P_total1 = calc_dnumber_isobaric(Tarr1, P0_total_1, T_seal)
ngas2, P_total2 = calc_dnumber_isobaric(Tarr2, P0_total_2, T_seal)
ngas3, P_total3 = calc_dnumber_isobaric(Tarr3, P0_total_3, T_seal)
ngas4, P_total4 = calc_dnumber_isobaric(Tarr4, P0_total_4, T_seal)

# pressure of target molecule
P_self1 = P_total1 * VMR  # [atm]
P_self2 = P_total2 * VMR
P_self3 = P_total3 * VMR
P_self4 = P_total4 * VMR

# Molecular number density array considering VMR (cgs)
nMolecule1 = VMR * ngas1
nMolecule2 = VMR * ngas2
nMolecule3 = VMR * ngas3
nMolecule4 = VMR * ngas4


# Wavenumber settings
def reverse_array(arr):
    return arr[::-1]


nu1 = 1e7 / reverse_array(wavd1)  # Wavenumber in ascending order
Trans_nu1 = reverse_array(Trans1)
Trans_nu2 = reverse_array(Trans2)
Trans_nu3 = reverse_array(Trans3)
Trans_nu4 = reverse_array(Trans4)

nu_min = nu1[0]
nu_max = nu1[-1]
nu_span = nu_max - nu_min

adjustrange = 0.2  # additional wavenumber for calculating the cross-section at the edge of wavelength range
gridboost = 10  # boosting factor of wavenumber resolution
valrange = 3
ndata1 = len(nu1)  # number of data points
nu_res_mean = nu_span / (ndata1 - 1)
polyx = np.linspace(0, nu_span, ndata1)  # wavenumber bin for polynomial

start_idx_nu = round(
    adjustrange / nu_res_mean
)  # Data points to cut from the shorter wavenumber region
Nx = round(ndata1 + start_idx_nu * 2)  # Data points including adjust range
Nx_boost = Nx * gridboost  # boosted datapoints


# Calculate the Line strength S(T)
def S_Tcalc(nu, S_0, T):
    logeS_0 = jnp.log(S_0)
    qr = mdb.qr_interp_lines(T)
    return line_strength(T, logeS_0, nu, mdb.elower, qr, Tref_original)


# polynominal fit function
def polynomial(a, b, c, d, x):
    return a * x**3 + b * x**2 + c * x + d


# generate the wavenumber&wavelength grid for cross-section
nu_grid, wav, res = wavenumber_grid(
    nu_min - adjustrange,
    nu_max + adjustrange,
    # jnp.max(wavd),
    Nx_boost,
    unit="cm-1",
    xsmode="premodit",
    wavelength_order="ascending",
)

sop_inst = SopInstProfile(nu_grid)

# Read the line database
mdb = MdbHitemp(
    ".database/CH4/",
    nurange=nu_grid,
    # crit =1E-30,
    gpu_transfer=False,  # Trueだと計算速度低下
    # with_error=True,
)  # for obtaining the error of each line

# Calculate the line index in the order of the line strength at T=twt
S_T = S_Tcalc(jnp.exp(mdb.nu_lines), mdb.line_strength_ref, Twt)
strline_ind = jnp.argsort(S_T)[::-1][0]

mdb_weak, nu_center_1st, mdb_voigt = create_mdbs(mdb, strline_ind)

# Create OpaPremodit instances for different temperature arrays
opa1 = create_opapremodit(mdb_weak, nu_grid, Tarr1)
opa2 = create_opapremodit(mdb_weak, nu_grid, Tarr2)
opa3 = create_opapremodit(mdb_weak, nu_grid, Tarr3)
opa4 = create_opapremodit(mdb_weak, nu_grid, Tarr4)


# Define the model for sampling parameters
def model_c(y1, y2, y3, y4):
    # Wavelength offset for each spectrum
    offrange = 0.05
    nu_offsets = [
        numpyro.sample(f"nu_offset{i+1}", dist.Uniform(-offrange, offrange))
        for i in range(4)
    ]

    # Line strength factor normalized by S(T) for the strongest line
    alphas = [numpyro.sample(f"alpha{i+1}", dist.Uniform(0.0, 5.0)) for i in range(4)]

    # Broadening parameters
    gamma_self = numpyro.sample("gamma_self", dist.Uniform(0.0, 0.2))  # [cm-1/atm]
    n = numpyro.sample("n", dist.Uniform(0, 2.0))

    # Polynomial coefficients
    coeffs = []
    for i in range(4):
        coeffs.append(
            {
                "a": numpyro.sample(
                    f"a{i+1}",
                    dist.Uniform(-valrange / nu_span**2, valrange / nu_span**2),
                ),
                "b": numpyro.sample(
                    f"b{i+1}", dist.Uniform(-valrange / nu_span, valrange / nu_span)
                ),
                "c": numpyro.sample(f"c{i+1}", dist.Uniform(-valrange, valrange)),
                "d": numpyro.sample(f"d{i+1}", dist.Uniform(0.0, 2.0)),
            }
        )

    # Gaussian noise for each spectrum
    sigmas = [numpyro.sample(f"sigma{i+1}", dist.Exponential(1.0e3)) for i in range(4)]

    # Calculate the polynomial functions
    polys = [
        polynomial(coeff["a"], coeff["b"], coeff["c"], coeff["d"], polyx)
        for coeff in coeffs
    ]

    # Calculate the Transmittance * polynomial for each spectrum
    trans_models = []
    for i, (
        nu_offset,
        alpha,
        polyfunc,
        tarr,
        p_total,
        p_self,
        n_molecule,
        opa,
    ) in enumerate(
        zip(
            nu_offsets,
            alphas,
            polys,
            [Tarr1, Tarr2, Tarr3, Tarr4],
            [P_total1, P_total2, P_total3, P_total4],
            [P_self1, P_self2, P_self3, P_self4],
            [nMolecule1, nMolecule2, nMolecule3, nMolecule4],
            [opa1, opa2, opa3, opa4],
        )
    ):
        trans_models.append(
            Trans_model_1Voigt(
                nu_offset,
                alpha,
                gamma_self,
                n,
                tarr,
                p_total,
                p_self,
                L,
                n_molecule,
                nu_grid,
                nu1,
                mdb_voigt,
                opa,
                sop_inst,
            )
            * polyfunc
        )

    # Sample the Transmittance * polynomial with Gaussian noise for each spectrum
    for i, (mu, sigma, y) in enumerate(zip(trans_models, sigmas, [y1, y2, y3, y4])):
        numpyro.sample(f"y{i+1}", dist.Normal(mu, sigma), obs=y)


print_parameters(
    Tarr1,
    Tarr2,
    Tarr3,
    Tarr4,
    Resolution,
    P_total1,
    P_total2,
    P_total3,
    P_total4,
    nu1,
    nu_span,
    valrange,
    Nx,
    mdb,
    strline_ind,
)

# Run mcmc
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)  # generate random numbers

# num_warmup, num_samples = 10, 10
# num_warmup, num_samples = 100, 50
num_warmup, num_samples = 500, 1000
# num_warmup, num_samples = 4000, 6000
kernel = NUTS(model_c, forward_mode_differentiation=False)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(
    rng_key_, y1=Trans_nu1, y2=Trans_nu2, y3=Trans_nu3, y4=Trans_nu4
)  # Run the MCMC samplers and collect samples

"""
initarray = "2.443e-03  6.700e-01  5.493e-02  5.118e-01 -8.885e-01 -4.832e-02 \
8.539e-02  9.966e-01  1.638e-03  7.537e-01  4.071e+01 -7.380e+00\
  2.223e-01  1.003e+00  3.607e-03  7.952e-01  8.019e+01 -1.407e+01\
  6.050e-01  9.955e-01  1.818e-03  7.519e-01  8.057e+01 -1.054e+01\
  3.579e-01  9.892e-01 \
  1.0e-3 1.0e-3 1.0e-3 1.0e-3"

initdata = list(map(float, initarray.split()))

initial = {
    "nu_offset1": initdata[0],
    "alpha1": initdata[1],
    "gamma_self": initdata[2],
    "n": initdata[3],
    "a1": initdata[4],
    "b1": initdata[5],
    "c1": initdata[6],
    "d1": initdata[7],
    "nu_offset2": initdata[8],
    "alpha2": initdata[9],
    "a2": initdata[10],
    "b2": initdata[11],
    "c2": initdata[12],
    "d2": initdata[13],
    "nu_offset3": initdata[14],
    "alpha3": initdata[15],
    "a3": initdata[16],
    "b3": initdata[17],
    "c3": initdata[18],
    "d3": initdata[19],
    "nu_offset4": initdata[20],
    "alpha4": initdata[21],
    "a4": initdata[22],
    "b4": initdata[23],
    "c4": initdata[24],
    "d4": initdata[25],
    "sigma1": initdata[26],
    "sigma2": initdata[27],
    "sigma3": initdata[28],
    "sigma4": initdata[29],
}

mcmc.run(
    rng_key_,
    y1=Trans_nu1,
    y2=Trans_nu2,
    y3=Trans_nu3,
    y4=Trans_nu4,
    init_params=initial,
)  # Run the MCMC samplers with initial parameters
"""

mcmc.print_summary()

# Save file name
savefilename = "Results/HMC/Model+poly/Multifit/240701-0703_162178-162193_Res00025_CH4VMR1-1path_Norm_Fremoved_HMC1500_sigma1e3_expdist_n0-2_gammaself00-02_al0-5_pval3-uniformdist_tgrad_wline_n-gamma-air_gb10_alcommon_xslpf_4dfit_refac"
yoffset = -0.6

plot_save_results(
    file1,
    file2,
    file3,
    file4,
    wavd1,
    Trans1,
    wavd2,
    Trans2,
    wavd3,
    Trans3,
    wavd4,
    Trans4,
    T_seal,
    VMR,
    P0_total_1,
    P0_total_2,
    P0_total_3,
    P0_total_4,
    P_total1,
    P_total2,
    P_total3,
    P_total4,
    polyx,
    polynomial,
    nu_center_1st,
    model_c,
    rng_key_,
    mcmc,
    savefilename,
    yoffset,
)


print("All done!")
