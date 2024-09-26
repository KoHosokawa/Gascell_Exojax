# %% [markdown]
# This script runs HMC including Teperature Gradients & weak lines for gamma self
# 2024/09/02 Created based on HMC_MultiVoigt_4dfit_gself_2408rev.py

##Read (& Show) the spectra data
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from HMC_defs_8data import (
    print_spectra_data,
    plot_spectra_data,
    create_mdbs_multi,
    create_opapremodit,
    print_parameters,
    plot_save_results_2cell,
    get_indices_for_wavelength_range,
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

# Road the spectra file(H2&He broadening cell)
direc5 = "../../実験/Data/240627/Corrected/"
direc6 = "../../実験/Data/240627/Corrected/"
direc7 = "../../実験/Data/240527/Corrected/"
direc8 = "../../実験/Data/240628/Corrected/"

file5 = "240627_wav1600-1630_res00025_0dbm_5kVW_ch1-H5path-CH4VMR01-P04W-rot30d_ch2-ref_SRQopt_T300K_wmask_ConM-X-10000_Y-16000_1352_1_TransRef-SpecCal0611wocell.txt"
file6 = "240627_wav1600-1630_res00025_0dbm_5kVW_ch1-H5path-CH4VMR01-P04W-rot30d_ch2-ref_SRQopt_T500K_wmask_ConM-X-2000_Y-2000_1744_1_TransRef-SpecCal0611wocell.txt"
file7 = "240527_laser1610_1630_res00025_0dbm_5kVW_ch1-H5path-CH4VMR01-P0-0423W-rot45deg_ch2-ref_SRQopt_432C_ConMirrorY-13000_1_TransRef-SpecCal0611wocell.txt"
file8 = "240628_wav1600-1630_res00025_0dbm_5kVW_ch1-H5path-CH4VMR01-P04W-rot30d_ch2-ref_SRQopt_1000K_wmask_ConM-X-5500_Y-12500_1642_1_TransRef-SpecCal0611wocell.txt"


# Store directories and filenames in lists
directories = [direc1, direc2, direc3, direc4, direc5, direc6, direc7, direc8]
files = [file1, file2, file3, file4, file5, file6, file7, file8]


# Load data
data_list = []
for i in range(len(directories)):
    delimiter = " " if i < 4 else ","
    data = np.genfromtxt(directories[i] + files[i], delimiter=delimiter, skip_header=0)
    data_list.append(data)


wav_array = data[:, 0]

# Example: Get the indices for the range from 1617.32 to 1617.57
start_wavelength = 1619.60
end_wavelength = 1619.85
start_index, end_index = get_indices_for_wavelength_range(
    wav_array, start_wavelength, end_wavelength
)
print(f"Start Index: {start_index}, End Index: {end_index}")


# Trim the data amount
start = start_index  # λ= 1617.32:6928, 1621.78:8712, 1628.235:11294
# dspan = 49
end = end_index
data_offset = -4000

# Initialize lists to store wavd and Trans data
wavd_array = []
trans_array = []

# Extract wavd and Trans, handling offset for the specific dataset
for i, data in enumerate(data_list):
    suffix = i + 1
    if suffix == 7:  # Apply offset for the specific dataset
        wavd = data[start + data_offset : end + data_offset, 0]
        trans = data[start + data_offset : end + data_offset, 1]
    else:
        wavd = data[start:end, 0]
        trans = data[start:end, 1]

    # Append the processed data to the lists
    wavd_array.append(wavd)
    trans_array.append(trans)


print_spectra_data(trans_array, wavd_array)
# plot_spectra_data(trans_array, wavd_array)
# plt.close()  # close the plot window

# Import the modules for Running HMC
from exojax.utils.constants import Tref_original, Tc_water  # [bar/atm]
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbHitemp
from exojax.spec.hitran import line_strength
from exojax.spec.specop import SopInstProfile
from Isobaric_Numdensity import calc_dnumber_isobaric
from Trans_model_MultiVoigt_HITEMP_nu_2408rev import Trans_model_MultiVoigt
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random

linenum = 2  # number of lines for Voigt fitting

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


# Temperature Setting(H2&He broadened cell)
Tarr5 = (
    np.array([24.0, 24.0, 24.0, 24.0, 23.9, 23.9, 23.7, 23.6]) + Tc_water
)  # CH3-10 of 2024/06/27 at 300K
Tarr6 = (
    np.array([231.0, 230.6, 230.2, 231.0, 230.9, 231.3, 233.2, 233.3]) + Tc_water
)  # CH3-10 of 2024/06/27 at 500K
Tarr7 = (
    np.array([429.6, 428.1, 429.5, 431.6, 431.6, 432.0, 434.0, 432.0]) + Tc_water
)  # CH3-10 of 2024/05/27 at 700K
Tarr8 = (
    np.array([720.1, 722.2, 728.1, 733.5, 732.9, 731.0, 729.4, 725.4]) + Tc_water
)  # CH3-10 of 2024/06/28 at 1000K

# create Temperature array
Tarrs = [Tarr1, Tarr2, Tarr3, Tarr4, Tarr5, Tarr6, Tarr7, Tarr8]
nspec = len(trans_array)

T_seal_1 = 23.6 + Tc_water
T_seal_2 = 21.4 + Tc_water
T_seal_array = np.array(
    [T_seal_1, T_seal_1, T_seal_1, T_seal_1, T_seal_2, T_seal_2, T_seal_2, T_seal_2]
)  # [K], The temperature at the time the cell was sealed. 23.6℃ for 100% cell, 21.4℃ for 10%cell

Twt = 1000  # Weighting Temperature

# Data infos
VMR1 = 1
VMR01 = 0.0981
VMR_array = np.array(
    [
        VMR1,
        VMR1,
        VMR1,
        VMR1,
        VMR01,
        VMR01,
        VMR01,
        VMR01,
    ]
)  # volume mixing ratio array

L_VMR1 = 49.7 * 1  # path length [cm]
L_VMR01 = 49.7 * 5

L_array = np.array(
    [
        L_VMR1,
        L_VMR1,
        L_VMR1,
        L_VMR1,
        L_VMR01,
        L_VMR01,
        L_VMR01,
        L_VMR01,
    ]
)

# Initialize pressures when the cell was sealed
P0_total_array = [0.423] * nspec  # Assuming the same initial pressure for simplicity

# Initialize lists to store results
ngas_array = [None] * nspec
P_total_array = [None] * nspec
P_self_array = [None] * nspec
nMolecule_array = [None] * nspec

# Loop through each dataset and perform calculations
for i in range(nspec):
    ngas_array[i], P_total_array[i] = calc_dnumber_isobaric(
        Tarrs[i], P0_total_array[i], T_seal_array[i]
    )
    P_self_array[i] = (
        P_total_array[i] * VMR_array[i]
    )  # Calculate pressure of target molecule [atm]
    nMolecule_array[i] = (
        VMR_array[i] * ngas_array[i]
    )  # Calculate molecular number density array considering VMR (cgs)


# Wavenumber settings
def reverse_array(arr):
    return arr[::-1]


# Initialize lists to store results
nu_array = [None] * nspec
trans_array_nu = [None] * nspec

# Loop through each dataset and perform calculations
for i in range(nspec):
    nu_array[i] = 1e7 / reverse_array(
        wavd_array[i]
    )  # Calculate wavenumber in ascending order
    trans_array_nu[i] = reverse_array(trans_array[i])  # Reverse the transmittance data


nu_min = nu_array[0][0]
nu_max = nu_array[0][-1]
nu_span = nu_max - nu_min


adjustrange = 0.2  # additional wavenumber for calculating the cross-section at the edge of wavelength range
gridboost = 10  # boosting factor of wavenumber resolution
valrange = 3
ndata1 = len(nu_array[0])  # number of data points
nu_res_mean = nu_span / (ndata1 - 1)
poly_nugrid = np.linspace(0, nu_span, ndata1)  # wavenumber bin for polynomial

start_idx_nu = round(
    adjustrange / nu_res_mean
)  # Data points to cut from the shorter wavenumber region
Nx = round(ndata1 + start_idx_nu * 2)  # Data points including adjust range
Nx_boost = Nx * gridboost  # boosted datapoints


# Calculate the Line strength S(T)
def S_Tcalc(nu, S_0, T):
    logeS_0 = jnp.log(S_0)
    qr = mdb.qr_interp_lines(T, Tref_original)
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
    # crit=1e-25,
    gpu_transfer=False,  # Trueだと計算速度低下
)  # for obtaining the error of each line


# Calculate the line index in the order of the line strength at T=twt
S_T = S_Tcalc(jnp.exp(mdb.nu_lines), mdb.line_strength_ref_original, Twt)
# Get indices where nu_lines values are within the specified range
valid_indices = jnp.where((mdb.nu_lines >= nu_min) & (mdb.nu_lines <= nu_max))[0]

# Sort the filtered indices based on line strength in descending order
S_T_filtered = S_T[valid_indices]
# strline_ind_array = jnp.argsort(S_T)[::-1][:linenum]
strline_ind_array = valid_indices[jnp.argsort(S_T_filtered)[::-1][:linenum]]
strline_ind_array_nu = jnp.sort(strline_ind_array)

mdb_weak, nu_center_voigt, mdb_voigt = create_mdbs_multi(mdb, strline_ind_array_nu)


# Initialize a list to store OpaPremodit instances
opa_array = []

# Create OpaPremodit instances for different temperature arrays
for Tarr in Tarrs:
    opa = create_opapremodit(mdb_weak, nu_grid, Tarr)
    opa_array.append(opa)


# Define the model for sampling parameters
def model_c(y1, y2, y3, y4, y5, y6, y7, y8):

    # Wavelength offset for each spectrum
    offrange = 0.05

    nu_offsets = [
        numpyro.sample(f"nu_offset{i+1}", dist.Uniform(-offrange, offrange))
        # numpyro.deterministic(f"nu_offset{i+1}", 0.0)
        for i in range(nspec)
    ]
    """
    # if you want to sepalate the each line offsets
    nu_offsets = jnp.array(
        [
            numpyro.sample(f"nu_offset{ns+1}_{ln+1}", dist.Uniform(-offrange, offrange))
            for ns in range(nspec)
            for ln in range(linenum)
        ]
    ).reshape(nspec, linenum)
    """
    # Line strength factor normalized by S(T) for the strongest line
    alphas = jnp.array(
        [
            numpyro.sample(f"alpha{ns+1}_{ln+1}", dist.Uniform(0.0, 3.0))
            for ns in range(nspec)
            for ln in range(linenum)
        ]
    ).reshape(nspec, linenum)

    # alphas = jnp.ones((nspec, linenum))  # fix alpha

    # Broadening parameters
    gamma_selfs = jnp.array(
        [
            numpyro.sample(f"gamma_self{i}", dist.Uniform(0.0, 0.2))
            # numpyro.sample(f"gamma_self{i}", dist.Uniform(0.060, 0.062))
            for i in range(1, 1 + linenum)
        ]
    )

    # Broadening parameters
    gamma_broads = jnp.array(
        [
            numpyro.sample(f"gamma_broad{i}", dist.Uniform(0.0, 0.2))
            # numpyro.sample(f"gamma_self{i}", dist.Uniform(0.060, 0.062))
            for i in range(1, 1 + linenum)
        ]
    )

    ns = jnp.array(
        [numpyro.sample(f"n{i}", dist.Uniform(0.0, 2)) for i in range(1, 1 + linenum)]
    )

    # Polynomial coefficients
    coeffs = []
    for i in range(nspec):
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
    """
    # Fixed values for parameters
    fixed_a = 0.0  # Replace with the desired fixed value
    fixed_b = 0.0  # Replace with the desired fixed value
    fixed_c = 0.0  # Replace with the desired fixed value
    fixed_d = 1.0  # Replace with the desired fixed value

    # Polynomial coefficients
    coeffs = []
    for i in range(nspec):
        coeffs.append(
            {
                "a": fixed_a,  # Using fixed value for 'a'
                "b": fixed_b,  # Using fixed value for 'b'
                "c": fixed_c,  # Using fixed value for 'c'
                "d": fixed_d,  # Using fixed value for 'd'
            }
        )

    # "d": fixed_d,  # Using fixed value for 'd'
    """
    # Gaussian noise for each spectrum
    sigmas = [
        numpyro.sample(f"sigma{i+1}", dist.Exponential(1.0e3)) for i in range(nspec)
    ]
    """
    fixed_sigma_value = 1e-10
    sigmas = [fixed_sigma_value for i in range(nspec)]
    """
    # Calculate the polynomial functions
    polys = [
        polynomial(coeff["a"], coeff["b"], coeff["c"], coeff["d"], poly_nugrid)
        for coeff in coeffs
    ]

    # Calculate the Transmittance * polynomial for each spectrum
    trans_models = []
    for i, (
        nu_offset,
        polyfunc,
        tarr,
        L,
        p_total,
        p_self,
        n_molecule,
        opa,
        nu_data,
        alphas_spec,
    ) in enumerate(
        zip(
            nu_offsets,
            polys,
            Tarrs,
            L_array,
            P_total_array,
            P_self_array,
            nMolecule_array,
            opa_array,
            nu_array,
            alphas,
        )
    ):

        trans_models.append(
            Trans_model_MultiVoigt(
                nu_offset,
                alphas_spec,
                gamma_broads,
                gamma_selfs,
                ns,
                tarr,
                p_total,
                p_self,
                L,
                n_molecule,
                nu_grid,
                nu_data,
                mdb_voigt,
                opa,
                sop_inst,
            )
            * polyfunc
        )

    # Sample the Transmittance * polynomial with Gaussian noise for each spectrum
    for i, (mu, sigma, y) in enumerate(
        zip(trans_models, sigmas, [y1, y2, y3, y4, y5, y6, y7, y8])
    ):
        numpyro.sample(f"y{i+1}", dist.Normal(mu, sigma), obs=y)


print_parameters(
    Tarrs,
    P_total_array,
    nu_span,
    valrange,
    Nx,
    mdb_voigt,
)

# Run mcmc
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)  # generate random numbers

# num_warmup, num_samples = 10, 10
# num_warmup, num_samples = 30, 70
num_warmup, num_samples = 1000, 2000
# num_warmup, num_samples = 2000, 3000
# num_warmup, num_samples = 2000, 3000
kernel = NUTS(model_c, forward_mode_differentiation=False)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)


def run_mcmc(mcmc, rng_key, trans_array_nu):
    # Prepare keyword arguments dynamically from the Trans_array
    kwargs = {f"y{i+1}": trans_nu for i, trans_nu in enumerate(trans_array_nu)}

    # Run MCMC model with unpacked keyword arguments
    mcmc.run(rng_key, **kwargs)


run_mcmc(mcmc, rng_key_, trans_array_nu)

mcmc.print_summary()

# %%
# Read the Wavelength range as str
wavmin_str = str(np.min(wavd)).replace(".", "")
wavmax_str = str(np.max(wavd)).replace(".", "")

# iteration number
Nitr = num_samples + num_warmup

# Save file name
savefilename = f"Results/HMC/Multifit/240701-0703_{wavmin_str}-{wavmax_str}_R00025_CH4VMR01-1_HMC{Nitr}_{linenum}V_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_{nspec}dfit"
# savefilename = f"Results/HMC/Multifit/Model_{wavmin_str}-{wavmax_str}_Res00025_CH4VMR1-1path_Norm_Fremoved_HMC{Nitr}_{linenum}Voigt-multi_sig1e+3_expdist_nuoff-01_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_{nspec}dfit"
# yoffset = -round((np.max(Trans3) - np.min(Trans3) + 0.1), 1)

plot_save_results_2cell(
    wavd_array,
    trans_array,
    T_seal_array,
    P0_total_array,
    P_total_array,
    VMR_array,
    poly_nugrid,
    polynomial,
    nu_center_voigt,
    model_c,
    rng_key_,
    mcmc,
    savefilename,
    mdb_voigt,
    nspec,
    linenum,
)


print("All done!")

# %%
