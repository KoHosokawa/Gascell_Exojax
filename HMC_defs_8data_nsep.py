# 2024/08/21 Updated: using corner.py for corner plots

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import vmap
from exojax.utils.constants import Tref_original
import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive

# from exojax.spec.hitran import gamma_hitran
from exojax.spec.opacalc import OpaPremodit
from datetime import datetime
from exojax.utils.constants import Patm

# from exojax.spec.opacalc import OpaPremodit
import numpy as np
import copy
import re
import scienceplots

plt.style.use(["science", "nature"])


import matplotlib.pyplot as plt


def print_spectra_data(trans_array, wavd_array):
    """
    Print the number of data points and wavelength ranges for each dataset in Trans_array and Wavd_array.

    Parameters:
    Trans_array : list of arrays containing transmittance data
    Wavd_array : list of arrays containing wavelength data
    """

    # Print the number of data points for each Trans
    trans_lengths = [len(trans) for trans in trans_array]
    print("Data points =", ", ".join(map(str, trans_lengths)))

    # Print the wavelength ranges for each wavd
    wavelength_ranges = [f"{wavd[0]}-{wavd[-1]} nm" for wavd in wavd_array]
    print("Wavelength Ranges =", ", ".join(wavelength_ranges))


def plot_spectra_data(trans_array, wavd_array):
    """
    Plot up to 8 spectra data sets on a single graph. Data is read from Trans_array and Wavd_array.

    Parameters:
    trans_array : list of arrays containing transmittance data
    wavd_array : list of arrays containing wavelength data
    """

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

    # Define colors and labels
    colors = ["C0", "C1", "C2", "C3", "C0", "C1", "C2", "C3"]
    labels = [f"Data{i+1}" for i in range(len(trans_array))]

    # Plot each dataset
    for i, (trans, wavd) in enumerate(zip(trans_array, wavd_array)):
        plt.plot(wavd, trans, ".-", alpha=1, color=colors[i % 8], label=labels[i])

    plt.xlabel("wavelength (nm)")
    plt.legend()
    plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
    ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
    ax.minorticks_on()
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling

    plt.show()
    plt.clf()


# Function to calculate indices for the given wavelength range
def get_indices_for_wavelength_range(wav_array, start_wavelength, end_wavelength):
    # Find the index closest to the starting wavelength
    start_index = np.abs(wav_array - start_wavelength).argmin()
    # Find the index closest to the ending wavelength
    end_index = np.abs(wav_array - end_wavelength).argmin() + 1

    return start_index, end_index


# Create the mdb copy and remove the strongest line
def create_mdbs(mdb, strline_ind):
    mdb_weak = copy.deepcopy(mdb)
    nu_center_voigt = mdb_weak.nu_lines[strline_ind]
    mask = mdb_weak.nu_lines != nu_center_voigt
    mdb_weak.apply_mask_mdb(mask)

    # Check if the mdb_weak has one less data point
    if len(mdb_weak.nu_lines) != len(mdb.nu_lines) - 1:
        raise ValueError(
            "mdb_weak does not have the correct number of data points after removing the strongest line."
        )

    # Create the mdb class only including the line for Voigt fitting
    mdb_voigt = copy.deepcopy(mdb)
    mdb_voigt.apply_mask_mdb(~mask)
    mdb_voigt.logsij0 = mdb_voigt.logsij0[strline_ind]

    # Check if the mdb_voigt contains only the strongest line
    if len(mdb_voigt.nu_lines) != 1:
        raise ValueError("mdb_voigt does not contain only the strongest line.")
    if mdb_voigt.nu_lines[0] != nu_center_voigt:
        raise ValueError(
            "The strongest line in mdb_voigt does not match nu_center_voigt."
        )
    return mdb_weak, nu_center_voigt, mdb_voigt


def create_mdbs_multi(mdb, strline_ind_array_nu):
    # Create the mdb copy and remove the strongest line
    mdb_weak = copy.deepcopy(mdb)

    # Ensure strline_ind is a 1D array
    strline_inds = np.array(strline_ind_array_nu).flatten()
    nu_centers = mdb_weak.nu_lines[strline_inds]

    # nu_centers = [mdb_weak.nu_lines[i] for i in strline_inds]

    # Create a mask: Set True for indices not included in strline_ind_array
    mask = np.ones_like(mdb_weak.nu_lines, dtype=bool)
    mask[strline_inds] = False  # Set positions corresponding to strline_ind to False

    # Apply the mask
    mdb_weak.apply_mask_mdb(mask)

    # Check if the mdb_weak has one less data point
    if len(mdb_weak.nu_lines) != len(mdb.nu_lines) - strline_inds.size:
        raise ValueError(
            "mdb_weak does not have the correct number of data points after removing the strongest line."
        )

    # Create the mdb class only including the line for Voigt fitting
    mdb_voigt = copy.deepcopy(mdb)
    mdb_voigt.apply_mask_mdb(~mask)
    strline_inds = np.atleast_1d(strline_inds)
    mdb_voigt.logsij0 = mdb_voigt.logsij0[strline_inds]

    # Check if the mdb_voigt contains only the strongest line
    if len(mdb_voigt.nu_lines) != strline_inds.size:
        raise ValueError("mdb_voigt does not contain only the strong lines.")
    if not np.all(np.isin(mdb_voigt.nu_lines, nu_centers)):
        raise ValueError("The strong line in mdb_voigt does not match nu_center_array.")
    return mdb_weak, nu_centers, mdb_voigt


# Define a function to create an OpaPremodit instance
def create_opapremodit(mdb_weak, nu_grid, tarr):
    return OpaPremodit(
        mdb=mdb_weak,
        nu_grid=nu_grid,
        diffmode=0,  # i-th Taylor expansion is used for the weight, default is 0.
        auto_trange=(np.min(tarr), np.max(tarr)),
    )  # opacity calculation


def print_parameters(Tarrs, P_total_array, nu_span, valrange, Nx, mdb_voigt):

    # Print pressure and temperature for each dataset
    for i, (P_total, Tarr) in enumerate(zip(P_total_array, Tarrs)):
        print(f"P{i+1} = {P_total}, T{i+1} = {Tarr}")

    # Print polynomial value ranges
    print(
        "polinomial value range a = ",
        -valrange / nu_span**2,
        valrange / nu_span**2,
        "b = ",
        valrange / nu_span,
        valrange / nu_span,
        "c = ",
        valrange,
        valrange,
    )

    # Print resolution and spectral data points
    print("Spectral Data point of model = " + str(Nx))

    # Print line centers at HITEMP with lambda unit conversion
    print("Line centers at HITEMP λ= " + str(1.0e7 / mdb_voigt.nu_lines) + "nm")

    # print
    for i in range(len(mdb_voigt.nu_lines)):
        print(
            "Voigt fitting Line center at HITEMP λ= "
            + str(1.0e7 / mdb_voigt.nu_lines[i])
            + "nm, gamma_self="
            + str(mdb_voigt.gamma_self[i])
            + ", gamma_air="
            + str(mdb_voigt.gamma_air[i])
            + ", n_air="
            + str(mdb_voigt.n_air[i])
        )


from exojax.utils.constants import Patm


def calculate_gammaL_nsep(
    P0_total, Tcenter, T_seal, VMR, predictions, mdb_voigt, linenum
):
    P_total = P0_total * Tcenter / T_seal
    P_self = P_total * VMR

    n_H2Hes = jnp.array([predictions[f"n_H2He{i}"] for i in range(1, linenum + 1)])
    n_selfs = jnp.array([predictions[f"n_self{i}"] for i in range(1, linenum + 1)])

    # Check if gamma_refs should be vectorized
    if any(f"gamma_ref{i}" in predictions for i in range(1, linenum + 1)):
        gamma_refs = jnp.array(
            [predictions[f"gamma_ref{i}"] for i in range(1, linenum + 1)]
        )
    else:
        gamma_refs = mdb_voigt.gamma_air

    # Check if gamma_selfs should be vectorized
    if any(f"gamma_self{i}" in predictions for i in range(1, linenum + 1)):
        gamma_selfs = jnp.array(
            [predictions[f"gamma_self{i}"] for i in range(1, linenum + 1)]
        )
    else:
        gamma_selfs = mdb_voigt.gamma_self

    # Handle the case where gamma_refs and gamma_selfs are arrays
    if gamma_refs.shape == () and gamma_selfs.shape == ():  # Scalars
        gammaL = vmap(gamma_hitran_nsep, in_axes=(0, 0, 0, None, None, None, None))(
            P_total, Tcenter, P_self, n_H2Hes, n_selfs, gamma_refs, gamma_selfs
        )

    else:  # Arrays
        gammaL = jnp.array(
            [
                gamma_hitran_nsep(
                    P_total,
                    Tcenter,
                    P_self,
                    n_H2Hes[i],
                    n_selfs[i],
                    gamma_refs[i],
                    gamma_selfs[i],
                )
                for i in range(linenum)
            ]
        )
    return gammaL


def gamma_hitran_nsep(P, T, Pself, n_broad, n_self, gamma_broad_ref, gamma_self_ref):
    """gamma factor by a pressure broadening.

    Args:
        P: pressure (bar)
        T: temperature (K)
        Pself: partial pressure (bar)
        n_broad: coefficient of the temperature dependence of the air-broadened halfwidth
        n_self: above self-broadened halfwidth
        gamma_broad_ref: gamma of the broadner
        gamma_self_ref: gamma self

    Returns:
        gamma: pressure gamma factor (cm-1)
    """
    Tref = Tref_original  # reference tempearture (K)
    gamma = (Tref / T) ** n_broad * (gamma_broad_ref * ((P - Pself) / Patm)) + (
        (Tref / T) ** n_self
    ) * gamma_self_ref * (Pself / Patm)

    return gamma


def calculate_gammaL_peratm(Tcenter, VMR, predictions, mdb_voigt, linenum):
    P_1atm = 1.0 * Patm  # [bar]
    P_self_1atm = P_1atm * VMR  # [bar]

    n_H2Hes = jnp.array([predictions[f"n_H2He{i}"] for i in range(1, linenum + 1)])
    n_selfs = jnp.array([predictions[f"n_self{i}"] for i in range(1, linenum + 1)])

    # Check if gamma_H2Hes should be vectorized
    if any(f"gamma_H2He{i}" in predictions for i in range(1, linenum + 1)):
        gamma_H2Hes = jnp.array(
            [predictions[f"gamma_H2He{i}"] for i in range(1, linenum + 1)]
        )
    else:
        gamma_H2Hes = mdb_voigt.gamma_air

    # Check if gamma_selfs should be vectorized
    if any(f"gamma_self{i}" in predictions for i in range(1, linenum + 1)):
        gamma_selfs = jnp.array(
            [predictions[f"gamma_self{i}"] for i in range(1, linenum + 1)]
        )
    else:
        gamma_selfs = mdb_voigt.gamma_self

    # Handle the case where gamma_H2Hes and gamma_selfs are arrays
    if gamma_H2Hes.shape == () and gamma_selfs.shape == ():  # Scalars
        gammaL = vmap(gamma_hitran_nsep, in_axes=(None, None, None, 0, 0, None, None))(
            P_1atm, Tcenter, P_self_1atm, n_H2Hes, n_selfs, gamma_H2Hes, gamma_selfs
        )
    else:  # Arrays
        gammaL_peratm = jnp.array(
            [
                gamma_hitran_nsep(
                    P_1atm,
                    Tcenter,
                    P_self_1atm,
                    n_H2Hes[i],
                    n_selfs[i],
                    gamma_H2Hes[i],
                    gamma_selfs[i],
                )
                for i in range(linenum)
            ]
        )
    return gammaL_peratm


# Print the input data
def print_results(
    P_total_array,
    Tcenters,
    keyarray,
    hpdi_values,
    median_value,
    gammaLkeyarray,
    linenum,
):
    nTcenter = len(Tcenters)
    print("Sampled parameters:", keyarray)
    for key in gammaLkeyarray:
        print(
            f"{key} = {median_value[key]:#.5g}, Lower: {hpdi_values[key][0]:#.5g}, Upper: {hpdi_values[key][1]:#.5g}"
        )


def calc_keyarray(
    T_seal_array,
    VMR,
    P0_total_array,
    Tcenters,
    poly_nugrid,
    polynomial,
    model_c,
    rng_key_,
    mcmc,
    mdb_voigt,
    nspec,
    linenum,
):
    nTcenter = len(Tcenters)
    posterior_sample = mcmc.get_samples()
    keyarray = list(posterior_sample.keys())
    yarray = [f"y{i+1}" for i in range(nspec)]
    Ykeyarray = keyarray + yarray
    pred = Predictive(model_c, posterior_sample, return_sites=Ykeyarray)
    # Automatically generate arguments like y1=None, y2=None, ...
    y_kwargs = {f"y{i+1}": None for i in range(nspec)}

    # Call the pred function, expanding the arguments with **
    predictions = pred(rng_key_, **y_kwargs)

    hpdi_values = {}
    median_value = {}

    gammaLkey = [
        f"{prefix}_{Tcenters[i]}_{j+1}"
        for prefix in ["gammaL", "gammaL-peratm"]
        for i in range(nTcenter)
        for j in range(linenum)
    ]
    gammaLadd_keyarray = keyarray + gammaLkey
    YgammaLadd_keyarray = gammaLadd_keyarray + yarray
    for key in YgammaLadd_keyarray:
        if key in gammaLkey:
            for i in range(nTcenter):
                gammaL_array = calculate_gammaL_nsep(
                    P0_total_array[i],
                    Tcenters[i],
                    T_seal_array[i],
                    VMR[i],
                    predictions,
                    mdb_voigt,
                    linenum,
                )
                gammaLperatm_array = calculate_gammaL_peratm(
                    Tcenters[i],
                    VMR[i],
                    predictions,
                    mdb_voigt,
                    linenum,
                )

                for j in range(linenum):
                    key = f"gammaL_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gammaL_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gammaL_array[j], axis=0)

                    key = f"gammaL-peratm_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gammaLperatm_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gammaLperatm_array[j], axis=0)

        else:
            hpdi_values[key] = hpdi(predictions[key], 0.9)  # 90% range

            # specific process for "y1~y8" keys
            if re.match(r"^y\d+$", key):
                median_value[key] = jnp.median(predictions[key], axis=0)
            else:
                median_value[key] = np.median(posterior_sample[key])
                exec(f"hpdi_{key} = [{hpdi_values[key][0]}, {hpdi_values[key][1]}]")
                exec(f"median_{key} = {median_value[key]}")

            polyfuncs = []
            for i in range(1, 1 + nspec):
                keys = [f"a{i}", f"b{i}", f"c{i}", f"d{i}"]

                # Check if all keys exist
                if all(key in median_value for key in keys):
                    coeffs = [median_value[key] for key in keys]

                # If the polynomial is fixed, those value for plot is defied as below
                else:
                    coeffs = [0.0, 0.0, 0.0, 1.0]

                polyfunc = polynomial(*coeffs, poly_nugrid)
                polyfuncs.append(polyfunc)

    return (
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLadd_keyarray,
        polyfuncs,
    )


def calc_keyarray_2cell(
    T_seal_array,
    VMR,
    P0_total_array,
    Tcenters,
    poly_nugrid,
    polynomial,
    model_c,
    rng_key_,
    mcmc,
    mdb_voigt,
    nspec,
    linenum,
):
    nTcenter = len(Tcenters)
    posterior_sample = mcmc.get_samples()
    keyarray = list(posterior_sample.keys())
    yarray = [f"y{i+1}" for i in range(nspec)]
    Ykeyarray = keyarray + yarray
    pred = Predictive(model_c, posterior_sample, return_sites=Ykeyarray)
    # Automatically generate arguments like y1=None, y2=None, ...
    y_kwargs = {f"y{i+1}": None for i in range(nspec)}

    # Call the pred function, expanding the arguments with **
    predictions = pred(rng_key_, **y_kwargs)

    hpdi_values = {}
    median_value = {}

    gammaLkey = [
        f"{prefix}_{Tcenters[i]}_{j+1}"
        for prefix in ["gammaL", "gself_Tatm", "gH2He_Tatm"]
        for i in range(nTcenter)
        for j in range(linenum)
    ]
    gammaLadd_keyarray = keyarray + gammaLkey
    YgammaLadd_keyarray = gammaLadd_keyarray + yarray
    for key in YgammaLadd_keyarray:
        if key in gammaLkey:
            for i in range(nTcenter):
                gammaL_array = calculate_gammaL_nsep(
                    P0_total_array[i],
                    Tcenters[i],
                    T_seal_array[i],
                    VMR[i],
                    predictions,
                    mdb_voigt,
                    linenum,
                )
                gself_Tatm_array = calculate_gammaL_peratm(
                    Tcenters[i],
                    1.0,
                    predictions,
                    mdb_voigt,
                    linenum,
                )

                gH2He_Tatm_array = calculate_gammaL_peratm(
                    Tcenters[i],
                    0.0,
                    predictions,
                    mdb_voigt,
                    linenum,
                )

                for j in range(linenum):
                    key = f"gammaL_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gammaL_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gammaL_array[j], axis=0)

                    key = f"gself_Tatm_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gself_Tatm_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gself_Tatm_array[j], axis=0)

                    key = f"gH2He_Tatm_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gH2He_Tatm_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gH2He_Tatm_array[j], axis=0)

        else:
            hpdi_values[key] = hpdi(predictions[key], 0.9)  # 90% range

            # specific process for "y1~y8" keys
            if re.match(r"^y\d+$", key):
                median_value[key] = jnp.median(predictions[key], axis=0)
            else:
                median_value[key] = np.median(posterior_sample[key])
                exec(f"hpdi_{key} = [{hpdi_values[key][0]}, {hpdi_values[key][1]}]")
                exec(f"median_{key} = {median_value[key]}")

            polyfuncs = []
            for i in range(1, 1 + nspec):
                keys = [f"a{i}", f"b{i}", f"c{i}", f"d{i}"]

                # Check if all keys exist
                if all(key in median_value for key in keys):
                    coeffs = [median_value[key] for key in keys]

                # If the polynomial is fixed, those value for plot is defied as below
                else:
                    coeffs = [0.0, 0.0, 0.0, 1.0]

                polyfunc = polynomial(*coeffs, poly_nugrid)
                polyfuncs.append(polyfunc)

    return (
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLadd_keyarray,
        polyfuncs,
    )


# Plot the spectra and fits
def plot_save_2cell(
    wavd_array,
    trans_array,
    nu_center_voigt,
    savefilename,
    hpdi_values,
    median_value,
    polyfuncs,
    offsets,
    nspec,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6 * (nspec / 2)))
    colors = ["C0", "C1", "C2", "C3"]

    linecenters = []

    for i in range(4):
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            color=colors[i],
            label=f"HMC Median Result {i+1}",
        )
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color=colors[i],
            label=f"90 \% area {i+1}",
        )
        ax.plot(
            wavd_array[i],
            trans_array[i] + offsets[i],
            ".",
            color="black",
            label=f"Measured spectra {i+1}, Yoffset = {offsets[i]:.2g}",
        )
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "--",
            linewidth=2,
            color=colors[i],
            label=f"Polynomial Component (median) {i+1}",
        )

        # Calclate the line center values
        offset_key = f"nu_offset{i+1}"
        if offset_key in median_value:
            linecenter = 1e7 / (nu_center_voigt + median_value[offset_key])
            linecenters.append(linecenter)
        else:
            linecenters.append(None)  # Handle missing offsets gracefully

        linemax = np.max(trans_array[i]) + 0.02
        linemin = np.min(trans_array[i]) - 0.02

        plt.vlines(
            linecenters[i],
            linemin + offsets[i],
            linemax + offsets[i],
            linestyles="--",
            linewidth=2,
            color="gray",
            label=f"Voigt fitted line center (median) {i+1}",
        )

    # Additional plot settings
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Intensity Ratio")
    plt.legend()
    plt.ylim(
        np.min(trans_array[3] + offsets[1] * 3) + offsets[1],
        np.max(trans_array[0]) - offsets[1] * 0.5,
    )
    plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
    ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
    ax.minorticks_on()
    plt.tick_params(labelsize=16)
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=10, ncol=2)
    plt.savefig(savefilename + "_spectra1-4.jpg", bbox_inches="tight")
    plt.close()

    # 2nd plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6 * (nspec / 2)))
    specoffset = 4

    for i in range(specoffset, 4 + specoffset):
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            color=colors[i - specoffset],
            label=f"HMC Median Result {i+1}",
        )
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color=colors[i - specoffset],
            label=f"90 \% area {i+1}",
        )
        ax.plot(
            wavd_array[i],
            trans_array[i] + offsets[i],
            ".",
            color="black",
            label=f"Measured spectra {i+1}, Yoffset = {offsets[i]:.2g}",
        )
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "--",
            linewidth=2,
            color=colors[i - specoffset],
            label=f"Polynomial Component (median) {i+1}",
        )

        # Calclate the line center values
        offset_key = f"nu_offset{i+1}"
        if offset_key in median_value:
            linecenter = 1e7 / (nu_center_voigt + median_value[offset_key])
            linecenters.append(linecenter)
        else:
            linecenters.append(None)  # Handle missing offsets gracefully

        linemax = np.max(trans_array[i]) + 0.02
        linemin = np.min(trans_array[i]) - 0.02

        plt.vlines(
            linecenters[i],
            linemin + offsets[i],
            linemax + offsets[i],
            linewidth=2,
            linestyles="--",
            color="gray",
            label=f"Voigt fitted line center (median) {i+1}",
        )

    # Additional plot settings
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Intensity Ratio")
    plt.legend()
    plt.ylim(
        np.min(trans_array[7] + offsets[5] * 3) + offsets[5],
        np.max(trans_array[4]) - offsets[5] * 0.5,
    )
    plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
    ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
    ax.minorticks_on()
    plt.tick_params(labelsize=16)
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=10, ncol=2)
    plt.savefig(savefilename + "_spectra5-8.jpg", bbox_inches="tight")
    plt.close()

    print("Spectra plot done!")


# Plot the spectra and fits
def plot_save_1cell(
    wavd_array,
    trans_array,
    nu_center_voigt,
    savefilename,
    hpdi_values,
    median_value,
    gammaLkeyarray,
    polyfuncs,
    offsets,
    nspec,
    linenum,
    Tcenters,
    mdb_voigt,
):
    import corner
    import pandas as pd

    # graph plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6 * (nspec / 2)))
    colors = ["C0", "C1", "C2", "C3"]

    linecenters = []

    for i in range(nspec):
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            color=colors[i],
            label=f"HMC Median Result {i+1}",
        )
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color=colors[i],
            label=f"90 \% area {i+1}",
        )
        ax.plot(
            wavd_array[i],
            trans_array[i] + offsets[i],
            ".",
            color="black",
            label=f"Measured spectra {i+1}, Yoffset = {offsets[i]:.2g}",
        )
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "--",
            linewidth=2,
            color=colors[i],
            label=f"Polynomial Component (median) {i+1}",
        )

        # Calclate the line center values
        linecenters = []

        for j in range(linenum):
            offset_key = f"nu_offset{i+1}"
            # offset_key = f"nu_offset{i+1}_{j+1}"  # if you sepalate the nu_offset
            # Add the nu_offset value to the corresponding nu_center_voigt value
            linecenter = 1e7 / (nu_center_voigt[j] + median_value[offset_key])
            linecenters.append(linecenter)

        linemax = np.max(trans_array[i]) + 0.02
        linemin = np.min(trans_array[i]) - 0.02

        plt.vlines(
            linecenters,
            linemin + offsets[i],
            linemax + offsets[i],
            linestyles="--",
            linewidth=2,
            color="gray",
            # label=f"Voigt fitted line center (median) {i+1}",
        )

    # Additional plot settings
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Intensity Ratio")
    plt.legend()
    plt.ylim(
        np.min(trans_array[nspec - 1] + offsets[1] * 3) + offsets[1],
        np.max(trans_array[0]) - offsets[1] * 0.5,
    )
    plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
    ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
    ax.minorticks_on()
    plt.tick_params(labelsize=16)
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=10, ncol=2)
    plt.savefig(savefilename + "_spectra.jpg", bbox_inches="tight")
    plt.close()

    print("Spectra plot done!")


def output_result_corner_pkl(
    nu_center_voigt,
    mcmc,
    savefilename,
    posterior_sample,
    keyarray,
    hpdi_values,
    median_value,
    gammaLkeyarray,
    Tcenters,
    mdb_voigt,
    wavd_array,
):
    import corner
    import pandas as pd

    # Output the results to a text file
    # Get the current date and time

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

        # Write the data from gammaLkeyarray
        for key in gammaLkeyarray:
            f.write(
                f"{key},{median_value[key]},{hpdi_values[key][0]},{hpdi_values[key][1]}\n"
            )

    # plot the distributions
    arviz.rcParams["plot.max_subplots"] = len(keyarray) ** 2
    arviz.plot_trace(
        mcmc, var_names=keyarray, backend_kwargs={"constrained_layout": True}
    )
    plt.savefig(savefilename + "_plotdist.jpg", bbox_inches="tight")
    # plt.show()
    plt.close()
    print("Distribution plot done!")

    # Save the posterior data
    import pickle

    with open(savefilename + "_post.pkl", "wb") as f:  # "wb"=binary mode(recommend)
        pickle.dump(posterior_sample, f)
    print("pkl file saved!")

    # Create the plot using `corner.corner`
    posterior_df = pd.DataFrame(posterior_sample)
    # Number of parameters to plot
    num_params = len(posterior_df.columns)

    figure = corner.corner(
        mcmc,
        figsize=(num_params, num_params),
        show_titles=True,
        label_kwargs={"fontsize": 16, "fontweight": "bold"},
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 10, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )
    plt.savefig(savefilename + "_cornerpy.jpg", bbox_inches="tight", dpi=50)
    # plt.show()
    plt.close()

    # Filter parameters to include only those starting with 'gamma' or 'n'
    filtered_var_names = [
        name
        for name in posterior_df.columns
        if name.startswith("gamma")
        or name.startswith("n")
        and not re.match(
            r"nu_offset\d*$", name
        )  # Excludes 'nu_offset' followed by any digits
    ]

    # Number of parameters to plot
    num_params_filter = len(filtered_var_names)

    # Create the plot using `corner.corner`
    figure = corner.corner(
        posterior_df[filtered_var_names],  # Use only the filtered parameters
        figsize=(num_params_filter, num_params_filter),
        show_titles=True,
        label_kwargs={
            "fontsize": 16,
            "fontweight": "bold",
        },  # Set font size and bold weight
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 10, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )

    plt.savefig(savefilename + "_cornerpy_gamma-n.jpg", bbox_inches="tight", dpi=100)
    # plt.show()
    plt.close()

    print("Corner plot done!")


def plot_save_results_2cell(
    wavd_array,
    trans_array,
    T_seal_array,
    P0_total_array,
    P_total_array,
    VMR,
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
):
    Tcenters = [300, 500, 700, 1000]
    yoffset1 = -round((np.max(trans_array[2]) - np.min(trans_array[2]) + 0.1), 1)
    yoffset2 = -round((np.max(trans_array[6]) - np.min(trans_array[6]) + 0.1), 1)
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
    (
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfuncs,
    ) = calc_keyarray_2cell(
        T_seal_array,
        VMR,
        P0_total_array,
        Tcenters,
        poly_nugrid,
        polynomial,
        model_c,
        rng_key_,
        mcmc,
        mdb_voigt,
        nspec,
        linenum,
    )

    print_results(
        P_total_array,
        Tcenters,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        linenum,
    )

    plot_save_2cell(
        wavd_array,
        trans_array,
        nu_center_voigt,
        savefilename,
        hpdi_values,
        median_value,
        polyfuncs,
        offsets,
        nspec,
    )

    output_result_corner_pkl(
        nu_center_voigt,
        mcmc,
        savefilename,
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        Tcenters,
        mdb_voigt,
        wavd_array,
    )


def plot_save_results_1cell(
    wavd_array,
    trans_array,
    T_seal_array,
    P0_total_array,
    P_total_array,
    VMR,
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
):
    Tcenters = [300, 500, 700, 1000]
    # Calculate the difference between the maximum and minimum value for each row,
    max_min_diffs = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(nspec)
    ]

    # Use the largest difference, add 0.1 to it, and set it as yoffset
    yoffset = -round(np.max(max_min_diffs) + 0.1, 1)
    offsets = [round(yoffset * i, 1) for i in range(nspec)]

    (
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfuncs,
    ) = calc_keyarray(
        T_seal_array,
        VMR,
        P0_total_array,
        Tcenters,
        poly_nugrid,
        polynomial,
        model_c,
        rng_key_,
        mcmc,
        mdb_voigt,
        nspec,
        linenum,
    )

    print_results(
        P_total_array,
        Tcenters,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        linenum,
    )

    plot_save_1cell(
        wavd_array,
        trans_array,
        nu_center_voigt,
        savefilename,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfuncs,
        offsets,
        nspec,
        linenum,
        Tcenters,
        mdb_voigt,
    )

    output_result_corner_pkl(
        nu_center_voigt,
        mcmc,
        savefilename,
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        Tcenters,
        mdb_voigt,
        wavd_array,
    )
