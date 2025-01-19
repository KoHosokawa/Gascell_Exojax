# 2024/11/5 Updated: reviced the plot format of corner and spectral fit, save the predictions for pkl file

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


# Define the function for finding the closest index with delta output
def find_closest_indices(nu_lines, target_wavenumbers):
    """Find the closest indices for each wavenumber in the target_wavenumbers list,
    along with the delta (difference) from the target wavenumber."""
    results = []
    for wn in target_wavenumbers:
        differences = np.abs(nu_lines - wn)
        min_diff = differences.min()
        closest_indices = np.where(differences == min_diff)[
            0
        ]  # Find all closest indices

        # Output all closest indices and deltas
        print(f"Target wavenumber: {wn}")
        for idx in closest_indices:
            delta = differences[idx]
            print(f"Closest index: {idx}, nu_lines: {nu_lines[idx]}, delta: {delta}")

        # Choose the first index (could customize this logic if needed)
        chosen_index = closest_indices[0]
        print(
            f"Chosen index: {chosen_index} (nu_lines: {nu_lines[chosen_index]}, delta: {differences[chosen_index]})"
        )

        results.append(chosen_index)
    return results


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
        -(valrange**3),
        valrange**3,
        "b = ",
        valrange**2,
        valrange**2,
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
    """
    gamma factor by a pressure broadening.

    Args:
        P: pressure (bar)
        T: temperature (K)
        Pself: partial pressure (bar)
        n_broad: coefficient of the temperature dependence of the air-broadened halfwidth
        n_self: above self-broadened halfwidth
        gamma_broad_ref: gamma of the broadner
        gamma_self_ref: gamma self

    Returns:
        gamma: pressure gamma factor by summing the self and broadner broadenning(cm-1/atm)
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
    posterior_sample = mcmc.get_samples()
    keyarray = list(posterior_sample.keys())
    yarray = [f"y{i+1}" for i in range(nspec)]
    Ykeyarray = keyarray + yarray
    pred = Predictive(model_c, posterior_sample, return_sites=Ykeyarray)
    """
    # Automatically generate arguments like y1=None, y2=None, ...
    y_kwargs = {f"y{i+1}": None for i in range(nspec)}

    # Call the pred function, expanding the arguments with **
    predictions = pred(rng_key_, **y_kwargs)
    """
    # Automatically generate a list of arguments with None for each spectrum
    y_args = [None for _ in range(nspec)]

    # Call the pred function, expanding the arguments with *
    predictions = pred(rng_key_, *y_args)

    hpdi_values = {}
    median_value = {}

    gammaLkey = [
        f"spec{i+1}_{prefix}_{Tcenters[i]}_{j+1}"
        for prefix in ["gammaL", "gself_Tatm", "gH2He_Tatm"]
        for i in range(nspec)
        for j in range(linenum)
    ]
    gammaLadd_keyarray = keyarray + gammaLkey
    YgammaLadd_keyarray = gammaLadd_keyarray + yarray
    for key in YgammaLadd_keyarray:
        if key in gammaLkey:
            for i in range(nspec):
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
                    key = f"spec{i+1}_gammaL_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gammaL_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gammaL_array[j], axis=0)

                    key = f"spec{i+1}_gself_Tatm_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gself_Tatm_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gself_Tatm_array[j], axis=0)

                    key = f"spec{i+1}_gH2He_Tatm_{Tcenters[i]}_{j+1}"
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
        predictions,
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
        f"spec{i+1}_{prefix}_{Tcenters[i]}_{j+1}"
        for prefix in ["gammaL", "gself_Tatm", "gH2He_Tatm"]
        for i in range(nspec)
        for j in range(linenum)
    ]
    gammaLadd_keyarray = keyarray + gammaLkey
    YgammaLadd_keyarray = gammaLadd_keyarray + yarray
    for key in YgammaLadd_keyarray:
        if key in gammaLkey:
            #calcurate the γ(P,T)at T=Tcenter, P(derived from T_seal, P_seal, T_center), VMR[spectral index]
            for i in range(nspec):
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
                    key = f"spec{i+1}_gammaL_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gammaL_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gammaL_array[j], axis=0)

                    key = f"spec{i+1}_gself_Tatm_{Tcenters[i]}_{j+1}"
                    hpdi_values[key] = hpdi(gself_Tatm_array[j], 0.9)  # 90% range
                    median_value[key] = np.median(gself_Tatm_array[j], axis=0)

                    key = f"spec{i+1}_gH2He_Tatm_{Tcenters[i]}_{j+1}"
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
        predictions,
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
    Tcenters,
):
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

        # Plot the median result
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            linewidth=2,
            color="C0",
            label=f"Median of HMC posterior" if is_last_iteration else None,
        )

        # Plot the HPDI (Highest Posterior Density Interval) area
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color="C0",
            label=f"90\% area" if is_last_iteration else None,
        )

        # Plot the polynomial component
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "-.",
            linewidth=2,
            color="C0",
            label=f"Polynomial component" if is_last_iteration else None,
        )

        # Calculate the line center if offset data is available
        offset_key = f"nu_offset{i+1}"
        if offset_key in median_value:
            linecenter = 1e7 / (nu_center_voigt + median_value[offset_key])
            linecenters.append(linecenter)
        else:
            linecenters.append(None)

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
        np.min(trans_array[3] + offsets[3]) - 0.5,
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
    plt.savefig(savefilename + "_spectra1-4_2.jpg", bbox_inches="tight")
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

        # Plot the median result
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            linewidth=2,
            color="C0",
            label=f"Median of HMC posterior" if is_last_iteration else None,
        )

        # Plot the HPDI (Highest Posterior Density Interval) area
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color="C0",
            label=f"90\% area" if is_last_iteration else None,
        )

        # Plot the polynomial component
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "-.",
            linewidth=2,
            color="C0",
            label=f"Polynomial component" if is_last_iteration else None,
        )

        # Calculate the line center if offset data is available
        offset_key = f"nu_offset{i+1}"
        if offset_key in median_value:
            linecenter = 1e7 / (nu_center_voigt + median_value[offset_key])
            linecenters.append(linecenter)
        else:
            linecenters.append(None)

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
        np.min(trans_array[7] + offsets[7]) - 0.25,
        np.max(trans_array[4] + offsets[4]) + 0.1,
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
    plt.savefig(savefilename + "_spectra5-8_2.jpg", bbox_inches="tight")
    plt.savefig("test2.jpg")
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
    polyfuncs,
    offsets,
    nspec,
    Tcenters,
):
    import corner
    import pandas as pd

    # graph plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 3 + 6 * (nspec)))
    linecenters = []
    
    for i in range(nspec):
        # Determine if this is the last iteration
        is_last_iteration = i == range(nspec)[-1]

        # Plot the measured spectra
        ax.plot(
            wavd_array[i],
            trans_array[i] + offsets[i],
            ".",
            markersize=10,
            color="black",
            label=f"Measured spectra" if is_last_iteration else None,
        )

        # Plot the median result
        ax.plot(
            wavd_array[i],
            median_value[f"y{i+1}"][::-1] + offsets[i],
            linewidth=2,
            color="C0",
            label=f"Median of HMC posterior" if is_last_iteration else None,
        )

        # Plot the HPDI (Highest Posterior Density Interval) area
        ax.fill_between(
            wavd_array[i],
            hpdi_values[f"y{i+1}"][0][::-1] + offsets[i],
            hpdi_values[f"y{i+1}"][1][::-1] + offsets[i],
            alpha=0.3,
            color="C0",
            label=f"90\% area" if is_last_iteration else None,
        )

        # Plot the polynomial component
        ax.plot(
            wavd_array[i],
            polyfuncs[i][::-1] + offsets[i],
            "-.",
            linewidth=2,
            color="C0",
            label=f"Polynomial component" if is_last_iteration else None,
        )

        # Calculate the line center if offset data is available
        offset_key = f"nu_offset{i+1}"
        if offset_key in median_value:
            linecenter = 1e7 / (nu_center_voigt + median_value[offset_key])
            linecenters.append(linecenter)
        else:
            linecenters.append(None)

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
        np.min(trans_array[-1] + offsets[-1]) - 0.7,
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
    plt.savefig(savefilename + "_spectra.jpg", bbox_inches="tight")
    plt.savefig("test.jpg")
    plt.close()

    print("Spectra plot done!")


def output_result_corner_pkl(
    nu_center_voigt,
    mcmc,
    savefilename,
    posterior_sample,
    predictions,
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
    # save posterior and predictions in 1 pkl file
    # Combine both dictionaries into a single dictionary
    combined_data = {
        "posterior_sample": posterior_sample,
        "predictions": predictions,
    }

    # Save the combined data into a pkl file
    with open(savefilename + "_postpred.pkl", "wb") as f:
        pickle.dump(combined_data, f)  # Save as a binary file
    print("pkl file saved!")

    # Create the whole plot using `corner.corner`
    posterior_df = pd.DataFrame(posterior_sample)
    # Number of parameters to plot
    num_params = len(posterior_df.columns)

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

    # Create the plot using `corner.corner` for gamma and n
    figure = corner.corner(
        posterior_df[filtered_var_names],  # Use only the filtered parameters
        figsize=(num_params_filter, num_params_filter),
        show_titles=True,
        label_kwargs={
            "fontsize": 22,
            "fontweight": "bold",
        },  # Set font size and bold weight
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 13, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )

    plt.savefig(savefilename + "_cornerpy_gamma-n.jpg", bbox_inches="tight", dpi=100)
    # plt.show()
    plt.close()

    figure = corner.corner(
        mcmc,
        figsize=(num_params, num_params),
        show_titles=True,
        label_kwargs={"fontsize": 22, "fontweight": "bold"},
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 13, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )
    plt.savefig(savefilename + "_cornerpy.jpg", bbox_inches="tight", dpi=50)
    # plt.show()
    plt.close()

    print("Corner plot done!")


def output_result_corner_pkl_Jlower(
    nu_center_voigt,
    mcmc,
    savefilename,
    posterior_sample,
    predictions,
    keyarray,
    hpdi_values,
    median_value,
    gammaLkeyarray,
    Tcenters,
    mdb_voigt,
    wavd_array,
    Jlowers,
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
                + ", J_lower="
                + str(Jlowers[i])
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
    # save posterior and predictions in 1 pkl file
    # Combine both dictionaries into a single dictionary
    combined_data = {
        "posterior_sample": posterior_sample,
        "predictions": predictions,
    }

    # Save the combined data into a pkl file
    with open(savefilename + "_postpred.pkl", "wb") as f:
        pickle.dump(combined_data, f)  # Save as a binary file
    print("pkl file saved!")

    # Create the plot using `corner.corner`
    posterior_df = pd.DataFrame(posterior_sample)
    # Number of parameters to plot
    num_params = len(posterior_df.columns)

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

    # Create the plot using `corner.corner` for gamma and n
    figure = corner.corner(
        posterior_df[filtered_var_names],  # Use only the filtered parameters
        figsize=(num_params_filter, num_params_filter),
        show_titles=True,
        label_kwargs={
            "fontsize": 22,
            "fontweight": "bold",
        },  # Set font size and bold weight
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 13, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )

    plt.savefig(savefilename + "_cornerpy_gamma-n.jpg", bbox_inches="tight", dpi=100)
    # plt.show()
    plt.close()

    figure = corner.corner(
        mcmc,
        figsize=(num_params, num_params),
        show_titles=True,
        label_kwargs={"fontsize": 22, "fontweight": "bold"},
        title_fmt=".2g",  # Specify the format for the titles
        title_kwargs={"fontsize": 13, "fontweight": "bold"},  # Specify the font size
        color="C0",
        quantiles=[0.10, 0.5, 0.90],
        smooth=1.0,
    )
    plt.savefig(savefilename + "_cornerpy.jpg", bbox_inches="tight", dpi=50)
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
    Tcenters,
):

    # Calculate the difference between the maximum and minimum value for each row,
    max_min_diffs1 = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(0, 4)
    ]
    max_min_diffs2 = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(4, 7)
    ]
    yoffset1 = -round(np.max(max_min_diffs1) + 0.4, 1)
    yoffset2 = -round(np.max(max_min_diffs2) + 0.3, 1)
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
        predictions,
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
        Tcenters,
    )

    output_result_corner_pkl(
        nu_center_voigt,
        mcmc,
        savefilename,
        posterior_sample,
        predictions,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        Tcenters,
        mdb_voigt,
        wavd_array,
    )


def plot_save_results_2cell_Jlower(
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
    Jlowers,
    Tcenters,
):

    # Calculate the difference between the maximum and minimum value for each row,
    max_min_diffs1 = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(0, 4)
    ]
    max_min_diffs2 = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(4, 7)
    ]
    yoffset1 = -round(np.max(max_min_diffs1) + 0.4, 1)
    yoffset2 = -round(np.max(max_min_diffs2) + 0.3, 1)
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
        predictions,
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
        Tcenters,
    )

    output_result_corner_pkl_Jlower(
        nu_center_voigt,
        mcmc,
        savefilename,
        posterior_sample,
        predictions,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        Tcenters,
        mdb_voigt,
        wavd_array,
        Jlowers,
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
    spec_ind,
    linenum,
    Tcenters,
):
    
    # Calculate the difference between the maximum and minimum value for each row,
    max_min_diffs = [
        np.max(trans_array[i]) - np.min(trans_array[i]) for i in range(nspec)
    ]

    # Use the largest difference, add 0.1 to it, and set it as yoffset
    yoffset = -round(np.max(max_min_diffs) + 0.5, 1)
    offsets = [round(yoffset * i, 1) for i in range(nspec)]

    (
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfuncs,
        predictions,
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
        polyfuncs,
        offsets,
        nspec,
        Tcenters,
        )

    output_result_corner_pkl(
        nu_center_voigt,
        mcmc,
        savefilename,
        posterior_sample,
        predictions,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        Tcenters,
        mdb_voigt,
        wavd_array,
    )
