# %% [markdown]
# This script runs HMC including Teperature Gradients & weak lines for gamma self, gammaH2+He
# n is separated to n_self and n_air
# alpha is common in same VMR cells (1 alphas for 1 line)
# 2024/09/13 Created based on HMC_MultiVoigt_4dfit_g-H2He_gself.py

##Read (& Show) the spectra data
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from HMC_defs_8data_nsep import (
    print_spectra_data,
    plot_spectra_data,
    create_mdbs_multi,
    create_opapremodit,
    print_parameters,
    plot_save_results_2cell,
    get_indices_for_wavelength_range,
    find_closest_indices,
)
from sklearn.metrics import mean_squared_error
from icecream import ic


# Road the spectra file
direc1 = "Data/240701/Corrected/"
direc2 = "Data/240701/Corrected/"
direc3 = "Data/240701/Corrected/"
direc4 = "Data/240703/Corrected/"

file1 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_300K_wmask_1103_1_AveNorm_fringe_removed.dat"
file2 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_500K_wmask_45dMt-X0V-Y75V_1322_1_AveNorm_fringe_removed.dat"
file3 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_700K_wmask_45dMt-X75V-Y0V_1515_1_AveNorm_fringe_removed.dat"
file4 = "240703_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_1000K_wmask_45dMt-150V-Y150V_1416_1_AveNorm_fringe_removed.dat"

# Road the spectra file(H2&He broadening cell)
direc5 = "Data/240627/Corrected/"
direc6 = "Data/240627/Corrected/"
direc7 = "Data/240527/Corrected/"
direc8 = "Data/240628/Corrected/"

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

    # Normalize the second column
    mean_value = np.mean(data[:, 1])  # Calculate mean of the second column
    data[:, 1] /= mean_value  # Normalize the second column

    data_list.append(data)


wav_array = data[:, 0]

# Example: Get the indices for the range from 1617.32 to 1617.57
start_wavelength = 1617.57
end_wavelength = 1617.67
start_index, end_index = get_indices_for_wavelength_range(
    wav_array, start_wavelength, end_wavelength
)
print(f"Start Index: {start_index}, End Index: {end_index}")

linenum = 1

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
from Trans_model_MultiVoigt_HITEMP_nu_nsep import Trans_model_MultiVoigt_nsep
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
    np.array([23.5, 23.6, 23.4, 23.5, 23.4, 23.4, 23.4, 23.3]) + Tc_water
)  # CH3-100 of 2024/07/01 at 300K
Tarr2 = (
    np.array([228.8, 227.7, 230.3, 231.5, 231.0, 231.7, 234.2, 219.6]) + Tc_water
)  # CH3-100 of 2024/07/01 at 500K
Tarr3 = (
    np.array([426.3, 426.4, 431.9, 433.3, 431.7, 431.8, 434.7, 413.5]) + Tc_water
)  # CH3-100 of 2024/07/01 at 700K
Tarr4 = (
    np.array([720.5, 720.0, 727.2, 734.3, 733.8, 731.4, 729.7, 725.2]) + Tc_water
)  # CH3-100 of 2024/07/03 at 1000K


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
P0_total_array = [0.423] * nspec  #the same initial pressure at sealing[bar]

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
valrange = 4
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
    logeS_0 = np.log(S_0)
    qr = mdb.qr_interp_lines(T, Tref_original)
    return line_strength(T, logeS_0, nu, mdb.elower, qr, Tref_original)


# polynominal fit function
def polynomial(a, b, c, d, x):
    return a * x**3 + b * x**2 + c * x + d


# generate the wavenumber&wavelength grid for cross-section
nu_grid, wav, res = wavenumber_grid(
    nu_min - adjustrange,
    nu_max + adjustrange,
    # np.max(wavd),
    Nx_boost,
    unit="cm-1",
    xsmode="premodit",
    wavelength_order="ascending",
)

nu_min = nu_array[0][0]
nu_max = nu_array[0][-1]
nu_span = nu_max - nu_min



sop_inst = SopInstProfile(nu_grid)

# Read the line database
mdb = MdbHitemp(
    "CH4",
    nurange=nu_grid,
    # crit=1e-25,
    gpu_transfer=False,  # Trueだと計算速度低下
    parfile = "database/06_HITEMP2020_2.par",
    #parfile = "database/06_HITEMP2020_2.par",
    #with_error=True,
)  # for obtaining the error of each line
#mdb.add_error()

# Calculate the line index in the order of the line strength at T=twt
S_T = S_Tcalc(mdb.nu_lines, mdb.line_strength_ref_original, Twt)
# Get indices where nu_lines values are within the specified range
valid_indices = np.where((mdb.nu_lines >= nu_min) & (mdb.nu_lines <= nu_max))[0]

# Sort the filtered indices based on line strength in descending order
S_T_filtered = S_T[valid_indices]
# strline_ind_array = np.argsort(S_T)[::-1][:linenum]
strline_ind_array = valid_indices[np.argsort(S_T_filtered)[::-1][:linenum]]
strline_ind_array_nu = np.sort(strline_ind_array)
mdb_weak, nu_center_voigt, mdb_voigt = create_mdbs_multi(mdb, strline_ind_array_nu)


# Initialize a list to store OpaPremodit instances
opa_array = []

# Create OpaPremodit instances for different temperature arrays
for Tarr in Tarrs:
    opa = create_opapremodit(mdb_weak, nu_grid, Tarr)
    opa_array.append(opa)


#####################################################
# Import necessary libraries for optimization
from scipy.optimize import minimize, least_squares


# Define the model function for least-squares fitting
def model_function(params, *args):
    # Unpack the arguments
    wavd_array, trans_array, Tarrs, L_array, P_total_array, P_self_array, nMolecule_array, opa_array, mdb_voigt, sop_inst = args

    # Extract parameters from the flat params array
    nspec = len(trans_array)

    # Parameters for each spectrum
    nu_offsets = params[:nspec]
    alphas = params[nspec : nspec + linenum]
    gamma_H2Hes = params[nspec + linenum : nspec + linenum *2]
    gamma_selfs = params[nspec + linenum *2: nspec + linenum *3]
    n_H2Hes = params[nspec + linenum *3 : nspec + linenum *4]
    n_selfs = params[nspec + linenum *4 : nspec + linenum *5]
    coeffs = params[nspec + linenum *5:].reshape(nspec, 4)

    # Calculate the modeled transmittance
    residuals = []
    for i in range(nspec):
        nu_data = 1e7 / wavd_array[i][::-1]  # Wavenumber in ascending order
        trans_data = trans_array[i][::-1]

        # Polynomial function
        polyfunc = polynomial(*coeffs[i], poly_nugrid)

        # Transmittance model
        trans_model = Trans_model_MultiVoigt_nsep(
            nu_offsets[i],
            alphas,
            gamma_H2Hes,
            gamma_selfs,
            n_H2Hes,
            n_selfs,
            Tarrs[i],
            P_total_array[i],
            P_self_array[i],
            L_array[i],
            nMolecule_array[i],
            nu_grid,
            nu_data,
            mdb_voigt,
            opa_array[i],
            sop_inst,
        )
        residuals.append((trans_model * polyfunc - trans_data).flatten())
    
    residuals = np.concatenate(residuals) #49 * 8array to list of 392 datas

    #return np.sum(np.square(residuals))
    return residuals 





# Initial guess for parameters
nspec = len(trans_array)

num_params_per_spec = 5 # nu_offset, polynomial coefficients
d1 = 1
d2 = 1
valrange = 4

#############################
init_params = np.concatenate([
    np.full(nspec, 0.),   # nu_offsets
    np.full(linenum, 1.),     # alphas 
    np.full(linenum * 2, 0.), # gamma_selfs and gamma_H2Hes 
    np.full(linenum * 2, 0.),# n_selfs and n_H2Hes 
    #np.full(linenum * 2, 0.), # gamma_selfs and gamma_H2Hes 
    #np.full(linenum * 2, 0.),# n_selfs and n_H2Hes 
    [0,0,0,d1],# Polynomial coefficients a~d inspec1
    [0,0,0,d1],
    [0,0,0,d1],
    [0,0,0,d1],
    [0,0,0,d2],
    [0,0,0,d2],
    [0,0,0,d2],
    [0,0,0,d2],
])
####################################

#optimization tolerance
tol = 1E-8

# Read the Wavelength range as str
wavmin_str = str(np.min(wavd)).replace(".", "")
wavmax_str = str(np.max(wavd)).replace(".", "")

# Save file name
savefilename = f"Results/Leastsquare/240527-0703_{wavmin_str}-{wavmax_str}_R00025_CH4VMR01-1_{linenum}V_lesq_xfgtol-{tol}_init_g0-n0_optlm_{nspec}dfit-norm"
#savefilename = f"Results/Leastsquare/240527-0703_{wavmin_str}-{wavmax_str}_R00025_CH4VMR01-1_{linenum}V_lesq_xfgtol-{tol}_init_g005-n05_optlm_{nspec}dfit-norm"
# yoffset = -round((np.max(Trans3) - np.min(Trans3) + 0.1), 1)


init_nu_offsets = init_params[:nspec]
init_alphas = init_params[nspec : nspec + linenum]
init_gamma_H2Hes = init_params[nspec + linenum : nspec + linenum *2]
init_gamma_selfs = init_params[nspec + linenum *2: nspec + linenum *3]
init_n_H2Hes = init_params[nspec + linenum *3 : nspec + linenum *4]
init_n_selfs = init_params[nspec + linenum *4 : nspec + linenum *5]
init_coeffs = init_params[nspec + linenum *5:].reshape(nspec, 4)

# Define bounds separately as lower and upper bounds for least square
lower_bounds = np.concatenate([
    np.full(nspec, -0.05),   # nu_offsets bounds
    np.full(linenum, 0),     # alphas bounds
    np.full(linenum * 2, 0), # gamma_selfs and gamma_H2Hes bounds (set 1)
    np.full(linenum * 2, -2),# gamma_selfs and gamma_H2Hes bounds (set 2)
    [-valrange,-valrange**2,-valrange**3,0],# Polynomial coefficients
    [-valrange,-valrange**2,-valrange**3,0],  
    [-valrange,-valrange**2,-valrange**3,0],
    [-valrange,-valrange**2,-valrange**3,0],
    [-valrange,-valrange**2,-valrange**3,0],
    [-valrange,-valrange**2,-valrange**3,0],
    [-valrange,-valrange**2,-valrange**3,0],
    [-valrange,-valrange**2,-valrange**3,0],
])

upper_bounds = np.concatenate([
    np.full(nspec, 0.05),    # nu_offsets bounds
    np.full(linenum, 5),     # alphas bounds
    np.full(linenum * 2, 0.2), # gamma_selfs and gamma_H2Hes bounds (set 1)
    np.full(linenum * 2, 2),  # gamma_selfs and gamma_H2Hes bounds (set 2)
    [valrange,valrange**2,valrange**3,2],    # Polynomial coefficients
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
    [valrange,valrange**2,valrange**3,2], 
])



print("Initial params")
print("nu_offsets: ",init_nu_offsets)
print("alphas: ",init_alphas)
print("gamma_H2Hes: ",init_gamma_H2Hes)
print("gamma_selfs: ",init_gamma_selfs)
print("n_H2Hes: ",init_n_H2Hes)
print("n_selfs: ",init_n_selfs)
print("coeffs: ",init_coeffs)

print("Upper Boundalies")
print(upper_bounds)
print("Lower Boundalies")
print(lower_bounds)

"""
#check the shape of residual
residuals_test = model_function(init_params, wavd_array, trans_array, Tarrs, L_array, P_total_array, P_self_array, nMolecule_array, opa_array, mdb_voigt, sop_inst,
    )
print(type(residuals_test))  # <class 'numpy.ndarray'> になればOK
print(residuals_test.shape)  # (N,) の形になればOK（Nはデータ数）
"""

#optimization

result = least_squares(
    model_function,
    init_params,
    args=(wavd_array, trans_array, Tarrs, L_array, P_total_array, P_self_array, nMolecule_array, opa_array, mdb_voigt, sop_inst),
    #bounds=(lower_bounds, upper_bounds),
    ftol = tol,
    xtol = tol,
    gtol = tol,
    verbose =2,
    method='lm'
    ) #trf

# Extract fitted parameters
fitted_params = result.x
print("Optimization successful:", result.success)
print("Optimization status:", result.status)

fit_nu_offsets = fitted_params[:nspec]
fit_alphas = fitted_params[nspec : nspec + linenum]
fit_gamma_H2Hes = fitted_params[nspec + linenum : nspec + linenum *2]
fit_gamma_selfs = fitted_params[nspec + linenum *2: nspec + linenum *3]
fit_n_H2Hes = fitted_params[nspec + linenum *3 : nspec + linenum *4]
fit_n_selfs = fitted_params[nspec + linenum *4 : nspec + linenum *5]
fit_coeffs = fitted_params[nspec + linenum *5:].reshape(nspec, 4)

print("Fitted parameters")
print("nu_offsets: ",fit_nu_offsets)
print("alphas: ",fit_alphas)
print("gamma_H2Hes: ",fit_gamma_H2Hes)
print("gamma_selfs: ",fit_gamma_selfs)
print("n_H2Hes: ",fit_n_H2Hes)
print("n_selfs: ",fit_n_selfs)
print("coeffs: ",fit_coeffs)



#%%





poly_trans_wav_list = []
polyfunc_wav_list = []
rmse_list = []
# Plot the results
for i in range(nspec):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    nu_data = 1e7 / wavd_array[i][::-1]
    trans_data = trans_array[i][::-1]
    coeffs = fitted_params[nspec + linenum *5 + 4 *i :nspec + linenum *5 + 4 *(i+1)]
    polyfunc = polynomial(*coeffs, poly_nugrid)
    trans_model = Trans_model_MultiVoigt_nsep(
        fitted_params[i],
        fitted_params[nspec : nspec + linenum],
        fitted_params[nspec + linenum : nspec + linenum *2],
        fitted_params[nspec + linenum *2: nspec + linenum *3],
        fitted_params[nspec + linenum *3 : nspec + linenum *4],
        fitted_params[nspec + linenum *4 : nspec + linenum *5],
        Tarrs[i],
        P_total_array[i],
        P_self_array[i],
        L_array[i],
        nMolecule_array[i],
        nu_grid,
        nu_data,
        mdb_voigt,
        opa_array[i],
        sop_inst,
    )
    poly_trans = trans_model * polyfunc
    poly_trans_wav = poly_trans[::-1]
    poly_trans_wav_list.append(poly_trans_wav)
    polyfunc_wav_list.append(polyfunc[::-1])


    #initial functions
    trans_model_init = Trans_model_MultiVoigt_nsep(
        init_params[i],
        init_params[nspec : nspec + linenum],
        init_params[nspec + linenum : nspec + linenum *2],
        init_params[nspec + linenum *2: nspec + linenum *3],
        init_params[nspec + linenum *3 : nspec + linenum *4],
        init_params[nspec + linenum *4 : nspec + linenum *5],
        Tarrs[i],
        P_total_array[i],
        P_self_array[i],
        L_array[i],
        nMolecule_array[i],
        nu_grid,
        nu_data,
        mdb_voigt,
        opa_array[i],
        sop_inst,
    )
    coeffs_init = init_params[nspec + linenum *5 + 4 *i :nspec + linenum *5 + 4 *(i+1)]
    polyfunc_init = polynomial(*coeffs_init, poly_nugrid)
    poly_trans_init = trans_model_init * polyfunc_init
    poly_trans_wav_init = poly_trans_init[::-1]
    plt.plot(wavd_array[i], poly_trans_wav_init,"-",color="gray", linewidth=2,label=f"Initial Model {i+1}")
    plt.plot(wavd_array[i], trans_array[i],".", color="k", linewidth=2,label=f"Observed Spectrum {i+1}")
    plt.plot(wavd_array[i], poly_trans_wav,"-", color="C0",linewidth=2,label=f"Fitted Model {i+1}")
    plt.plot(wavd_array[i], polyfunc_wav_list[i],
        "--",
        color="C0",
        linewidth=2,
        label=f"Polynomial component" )

    plt.grid(which="major", axis="both", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=24)
    plt.xlabel("Wavelength (nm)", fontsize=36, labelpad=20)
    plt.ylabel("Intensity Ratio", fontsize=36, labelpad=20)
    plt.tick_params(labelsize=28)
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling



    plt.savefig(savefilename+"spec"+str(i+1)+".jpg", bbox_inches="tight")
    plt.close()

    #RMSE calcuration
    rmse = np.sqrt(mean_squared_error(trans_data, trans_model * polyfunc))
    print('RMSE in spec'+str(i+1)+': {:.3g}'.format(rmse))
    rmse_list.append(rmse)






# %%


# Define the representive temperature array
Tcenters = [297, 500, 700, 1000, 297, 500, 700, 1000]
#Tcenters = [297, 503, 702, 1001, 297, 505, 704, 1001]

 # Calculate the difference between the maximum and minimum value for each row,
max_min_diffs1 = [
    np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(0, 4)
]
max_min_diffs2 = [
    np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(4, 7)
]
yoffset1 = -round(np.max(max_min_diffs1) + 0.4, 1)
yoffset2 = -round(np.max(max_min_diffs2) + 0.4, 1)
offsets = [
    0,
    yoffset1,
    yoffset1 * 2,
    yoffset1 * 3,
    0,
    yoffset2,
    yoffset2 * 2,
    yoffset2 * 3,
]


######################################
# Plot the spectra and fits

# First plot for spectra 1-4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6 * (nspec / 2)))
linecenters = []
    
for i in range(4):
    # Determine if this is the last iteration
    is_last_iteration = i == range(4)[-1]

    # Plot the measured spectra
    ax.plot(
        wavd_array[i],
        trans_array[i] + offsets[i],
        ".",
        markersize=10,
        color="black",
        label=f"Measured spectra" if is_last_iteration else None,
    )

    # fitted model
    plt.plot(wavd_array[i], poly_trans_wav_list[i] + offsets[i], linewidth=2,
        color="C0", label=f"Fitted Model {i+1}" if is_last_iteration else None,)

    # Plot the polynomial component
    plt.plot(wavd_array[i], polyfunc_wav_list[i] + offsets[i],
        "-.",
        linewidth=2,
        color="C0",
        label=f"Polynomial component" if is_last_iteration else None,
    )

    # Calculate the line center if offset data is available
    linecenter = 1e7 / (nu_center_voigt + fit_nu_offsets[i])
    linecenters.append(linecenter)

    linespan = (np.max(trans_array[i]) - np.min(trans_array[i])) * 0.3
    linemin = np.max(trans_array[i])
    linemax = linemin + linespan

    # Plot the Voigt fitted line center
    plt.vlines(
        linecenters[i],
        linemin + offsets[i],
        linemax + offsets[i],
        linestyles="--",
        linewidth=2,
        color="gray",
        label=f"Target line centers" if is_last_iteration else None,
    )

    y_max = np.max(
        trans_array[i] + offsets[i]
    )  # Get the max y value for positioning

    ax.text(
        0.05,
        y_max + 0.2,  # Adjust position slightly above the graph
        f"T = {Tcenters[i]} K",
        fontsize=34,
        ha="left",
        va="top",
        transform=ax.get_yaxis_transform(),  # Ensure the text is relative to the y-axis
    )

# Configure plot settings
plt.ylim(
    np.min(trans_array[3] + offsets[3]) - 0.7,
    np.max(trans_array[0] + offsets[0]) + 0.3,
)
plt.xlabel("Wavelength (nm)", fontsize=36, labelpad=20)
plt.ylabel("Intensity Ratio", fontsize=36, labelpad=20)
plt.tick_params(labelsize=28)
plt.title(
    rf"$\text{{CH}}_4$ 100\%",
    fontsize=50,
    pad=20,
)

ax.get_xaxis().get_major_formatter().set_useOffset(
    False
)  # To avoid exponential labeling
# plt.grid(which="major", axis="both", linestyle="--", alpha=0.5)
plt.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=24)
plt.savefig(savefilename + "_spectra1-4.jpg", bbox_inches="tight")
plt.savefig("test.jpg")
plt.close()


# Second plot for spectra 5-8
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6 * (nspec / 2)))
specoffset = 4

for i in range(specoffset, 4 + specoffset):
    is_last_iteration = i == range(specoffset + 4)[-1]

    # Plot the measured spectra
    ax.plot(
        wavd_array[i],
        trans_array[i] + offsets[i],
        ".",
        markersize=10,
        color="black",
        label=f"Measured spectra" if is_last_iteration else None,
    )


    # fitted model
    plt.plot(wavd_array[i], poly_trans_wav_list[i] + offsets[i], linewidth=2,
        color="C0", label=f"Fitted Model" if is_last_iteration else None,)

    # Plot the polynomial component
    plt.plot(wavd_array[i], polyfunc_wav_list[i] + offsets[i],
        "-.",
        linewidth=2,
        color="C0",
        label=f"Polynomial component" if is_last_iteration else None,
    )


    # Calculate the line center if offset data is available
    linecenter = 1e7 / (nu_center_voigt + fit_nu_offsets[i])
    linecenters.append(linecenter)

    linespan = (np.max(trans_array[i]) - np.min(trans_array[i])) * 0.3
    linemin = np.max(trans_array[i])
    linemax = linemin + linespan


    # Plot the Voigt fitted line center
    plt.vlines(
        linecenters[i],
        linemin + offsets[i],
        linemax + offsets[i],
        linestyles="--",
        linewidth=2,
        color="gray",
        label=f"Target line centers" if is_last_iteration else None,
    )

    # Add text "T = 1000 K" in the upper left corner for each plot
    y_max = np.max(
        trans_array[i] + offsets[i]
    )  # Get the max y value for positioning

    ax.text(
        0.05,
        y_max + 0.07,  # Adjust position slightly above the graph
        f"T = {Tcenters[i]} K",
        fontsize=34,
        ha="left",
        va="top",
        transform=ax.get_yaxis_transform(),  # Ensure the text is relative to the y-axis
    )

# Configure plot settings
plt.ylim(
    np.min(trans_array[7] + offsets[7]) - 0.7,
    np.max(trans_array[4] + offsets[4]) + 0.3,
)
plt.xlabel("Wavelength (nm)", fontsize=36, labelpad=20)
plt.ylabel("Intensity Ratio", fontsize=36, labelpad=20)
plt.tick_params(labelsize=28)
ax.get_xaxis().get_major_formatter().set_useOffset(
    False
)  # To avoid exponential labeling
plt.title(
    rf"$\text{{CH}}_4$ 10\% + $\text{{H}}_2$\&He 90\%",
    fontsize=50,
    pad=20,
)

# plt.grid(which="major", axis="both", linestyle="--", alpha=0.5)
plt.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=24)
plt.savefig(savefilename + "_spectra5-8.jpg", bbox_inches="tight")
plt.savefig("test2.jpg")
plt.close()
print("Spectra plot done!")


from datetime import datetime

#output the result to txt file
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nu_center_voigt_flat = np.array(nu_center_voigt).flatten()

with open(f"{savefilename}_results.txt", "w") as f:
    # Write the timestamp header
    f.write(f"Recorded Date and Time: {current_time}\n")

    # Write Tcenters as a comma-separated list
    f.write("Tcenters: " + ",".join(map(str, Tcenters)) + "\n")
    f.write(f"Fitted wavelength range: {wavd_array[0][0]}-{wavd_array[0][-1]}\n")

    f.write("Voigt nu centers: " + ",".join(map(str, nu_center_voigt)) + "\n")
    f.write(
        "Voigt wavelength centers: "
        + ",".join(map(str, 1e7 / nu_center_voigt_flat))
        + "\n"
    )
    for i in range(len(nu_center_voigt)):
        f.write(
            "Voigt Line center "
            + str(i + 1)
            + " in HITEMP:λ= "
            + str(1.0e7 / mdb_voigt.nu_lines[i])
            + "nm, gamma_self="
            + str(mdb_voigt.gamma_self[i])
            + ", gamma_air="
            + str(mdb_voigt.gamma_air[i])
            + ", n_air="
            + str(mdb_voigt.n_air[i])
            + ", E_lower="
            + str(mdb_voigt.elower[i])
            + "\n"
        )
    f.write("\n")


    f.write("Initial params\n")
    f.write("nu_offsets: " + " ".join(map(str, init_nu_offsets)) + "\n")
    f.write("alphas: " + " ".join(map(str, init_alphas)) + "\n")
    f.write("gamma_H2Hes: " + " ".join(map(str, init_gamma_H2Hes)) + "\n")
    f.write("gamma_selfs: " + " ".join(map(str, init_gamma_selfs)) + "\n")
    f.write("n_H2Hes: " + " ".join(map(str, init_n_H2Hes)) + "\n")
    f.write("n_selfs: " + " ".join(map(str, init_n_selfs)) + "\n")
    f.write("coeffs: " + str(init_coeffs) + "\n")  # 2D配列は str() で書き出す

    # f.write("\nUpper Boundaries\n")
    # f.write(str(upper_bounds) + "\n")
    # f.write("Lower Boundaries\n")
    # f.write(str(lower_bounds) + "\n")

    f.write("\nOptimization status: " + str(result.status) + "\n")

    f.write("Parameters fitting results\n")
    f.write("nu_offsets: " + " ".join(map(str, fit_nu_offsets)) + "\n")
    f.write("alphas: " + " ".join(map(str, fit_alphas)) + "\n")
    f.write("gamma_H2Hes: " + " ".join(map(str, fit_gamma_H2Hes)) + "\n")
    f.write("gamma_selfs: " + " ".join(map(str, fit_gamma_selfs)) + "\n")
    f.write("n_H2Hes: " + " ".join(map(str, fit_n_H2Hes)) + "\n")
    f.write("n_selfs: " + " ".join(map(str, fit_n_selfs)) + "\n")
    f.write("coeffs: " + str(fit_coeffs) + "\n")  # 2D配列は str() で書き出す

    f.write("\nRMSE for each spec: " + " ".join(map(str, rmse_list)) + "\n")
f.close()


print("All done!")

# %%
