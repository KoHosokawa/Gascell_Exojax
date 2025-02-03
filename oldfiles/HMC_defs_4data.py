import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from exojax.utils.constants import Tref_original
import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from exojax.spec.hitran import gamma_hitran
from exojax.spec.opacalc import OpaPremodit

# from exojax.spec.opacalc import OpaPremodit
import numpy as np
import copy


def print_spectra_data(wavd1, Trans1, wavd2, Trans2, wavd3, wavd4):
    print("Data points = " + str(len(Trans1)) + ", " + str(len(Trans2)))
    print(
        "Wavelength Ranges = ",
        wavd1[0],
        "-",
        wavd1[-1],
        "nm, ",
        wavd2[0],
        "-",
        wavd2[-1],
        "nm",
        wavd3[0],
        "-",
        wavd3[-1],
        "nm, ",
        wavd4[0],
        "-",
        wavd4[-1],
        "nm",
    )


def plot_spectra_data(wavd1, Trans1, wavd2, Trans2, wavd3, Trans3, wavd4, Trans4):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    plt.plot(wavd1, Trans1, ".-", alpha=1, color="C0", label="Data1")
    plt.plot(wavd2, Trans2, ".-", alpha=1, color="C2", label="Data2")
    plt.plot(wavd3, Trans3, ".-", alpha=1, color="C1", label="Data3")
    plt.plot(wavd4, Trans4, ".-", alpha=1, color="C3", label="Data4")
    plt.xlabel("wavelength $\AA$")
    plt.legend()
    plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
    ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
    ax.minorticks_on()
    ax.get_xaxis().get_major_formatter().set_useOffset(
        False
    )  # To avoid exponential labeling

    plt.show()
    plt.clf()


# Create the mdb copy and remove the strongest line
def create_mdbs(mdb, strline_ind):
    mdb_weak = copy.deepcopy(mdb)
    nu_center_1st = mdb_weak.nu_lines[strline_ind]
    mask = mdb_weak.nu_lines != nu_center_1st
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
    if mdb_voigt.nu_lines[0] != nu_center_1st:
        raise ValueError(
            "The strongest line in mdb_voigt does not match nu_center_1st."
        )
    return mdb_weak, nu_center_1st, mdb_voigt


# Define a function to create an OpaPremodit instance
def create_opapremodit(mdb, nu_grid, tarr):
    return OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=0,  # i-th Taylor expansion is used for the weight, default is 0.
        auto_trange=(np.min(tarr), np.max(tarr)),
    )  # opacity calculation


def print_parameters(
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
):
    print("Tref =", Tref_original)
    print("P1 = ", P_total1, "T1 = ", Tarr1)
    print("P2 = ", P_total2, "T2 = ", Tarr2)
    print("P3 = ", P_total3, "T3 = ", Tarr3)
    print("P4 = ", P_total4, "T4 = ", Tarr4)

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
    print(
        "Model Resolution: R > "
        + str(nu1[0] / Resolution)
        + ",Spectral Data point of model = "
        + str(Nx)
    )
    print("Line centers at HITEMP λ= " + str(1.0e7 / mdb.nu_lines) + "nm")
    print(
        "Strongest Line center at HITEMP λ= "
        + str(1.0e7 / mdb.nu_lines[strline_ind])
        + "nm, gamma_self="
        + str(mdb.gamma_self[strline_ind])
        + ", gamma_air="
        + str(mdb.gamma_air[strline_ind])
        + ", n_air="
        + str(mdb.n_air[strline_ind])
    )


# Print the input data
def print_results(
    file1,
    file2,
    file3,
    file4,
    P_total1,
    P_total2,
    P_total3,
    P_total4,
    Tcenter1,
    Tcenter2,
    Tcenter3,
    Tcenter4,
    keyarray,
    hpdi_values,
    median_value,
    gammaLkeyarray,
):
    print("input file: \n\t", file1, "\n\t", file2, "\n\t", file3, "\n\t", file4)
    print("sampled parameter = ", keyarray)
    print(
        "gammaL at T = ",
        Tcenter1,
        "K, P = {:#.3g}".format(P_total1),
        "bar = ",
        median_value["gammaL_1"],
    )
    print(
        "gammaL at T = ",
        Tcenter2,
        "K, P = {:#.3g}".format(P_total2),
        "bar = ",
        median_value["gammaL_2"],
    )
    print(
        "gammaL at T = ",
        Tcenter3,
        "K, P = {:#.3g}".format(P_total3),
        "bar = ",
        median_value["gammaL_3"],
    )
    print(
        "gammaL at T = ",
        Tcenter4,
        "K, P = {:#.3g}".format(P_total4),
        "bar = ",
        median_value["gammaL_4"],
    )

    for key in gammaLkeyarray:
        # print(str(key)+ "= {:#.3g} +{:#.3g} -{:#.3g}".format(median_value[key],hpdi_values[key][1]-median_value[key],median_value[key]-hpdi_values[key][0]))
        print(
            str(key)
            + "= {:#.5g}, Lower: {:#.5g}, Upper: {:#.5g}".format(
                median_value[key], hpdi_values[key][0], hpdi_values[key][1]
            )
        )


def plot_save_results(
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
):
    def calc_keyarray(
        T_seal,
        VMR,
        P0_total_1,
        P0_total_2,
        P0_total_3,
        P0_total_4,
        polyx,
        polynomial,
        model_c,
        rng_key_,
        mcmc,
    ):
        # Temperature setting for calculating gammma(T,P)
        Tcenter1 = 300
        Tcenter2 = 500
        Tcenter3 = 700
        Tcenter4 = 1000

        posterior_sample = mcmc.get_samples()
        keyarray = list(posterior_sample.keys())
        Ykeyarray = keyarray + ["y1", "y2", "y3", "y4"]
        print("sampled parameter = ", keyarray)
        pred = Predictive(
            model_c, posterior_sample, return_sites=Ykeyarray
        )  # create the dictionary for all sample sites
        predictions = pred(rng_key_, y1=None, y2=None, y3=None, y4=None)

        # Calculate HPDI and median values as "hpdi_~" and "median_~"
        hpdi_values = {}
        median_value = {}

        Y_gammaLkeyarray = keyarray + [
            "y1",
            "y2",
            "y3",
            "y4",
            "gammaL_1",
            "gammaL_2",
            "gammaL_3",
            "gammaL_4",
        ]
        gammaLkeyarray = keyarray + ["gammaL_1", "gammaL_2", "gammaL_3", "gammaL_4"]

        def calculate_gammaL(P0_total, Tcenter, T_seal, VMR, predictions):
            P_total = P0_total * Tcenter / T_seal
            P_self = P_total * VMR
            gammaL = gamma_hitran(
                P_total,
                Tcenter,
                P_self,
                predictions["n"],
                0,
                predictions["gamma_self"],
            )
            return gammaL

        gammaL_parameters = {
            "gammaL_1": (P0_total_1, Tcenter1),
            "gammaL_2": (P0_total_2, Tcenter2),
            "gammaL_3": (P0_total_3, Tcenter3),
            "gammaL_4": (P0_total_4, Tcenter4),
        }

        for key in Y_gammaLkeyarray:
            if key in gammaL_parameters:
                P0_total, Tcenter = gammaL_parameters[key]
                gammaL = calculate_gammaL(P0_total, Tcenter, T_seal, VMR, predictions)
                hpdi_values[key] = hpdi(gammaL, 0.9)
                median_value[key] = np.median(gammaL, axis=0)
            else:
                hpdi_values[key] = hpdi(predictions[key], 0.9)  # 90% range

                if key in [
                    "y1",
                    "y2",
                    "y3",
                    "y4",
                ]:  # avoid y since its only array shape
                    median_value[key] = jnp.median(predictions[key], axis=0)
                else:
                    median_value[key] = np.median(posterior_sample[key])
                    exec(f"hpdi_{key} = [{hpdi_values[key][0]}, {hpdi_values[key][1]}]")
                    exec(f"median_{key} = {median_value[key]}")

        # Plot polynomial components
        polyfunc1 = polynomial(
            median_value["a1"],
            median_value["b1"],
            median_value["c1"],
            median_value["d1"],
            polyx,
        )
        polyfunc2 = polynomial(
            median_value["a2"],
            median_value["b2"],
            median_value["c2"],
            median_value["d2"],
            polyx,
        )
        polyfunc3 = polynomial(
            median_value["a3"],
            median_value["b3"],
            median_value["c3"],
            median_value["d3"],
            polyx,
        )
        polyfunc4 = polynomial(
            median_value["a4"],
            median_value["b4"],
            median_value["c4"],
            median_value["d4"],
            polyx,
        )

        return (
            Tcenter1,
            Tcenter2,
            Tcenter3,
            Tcenter4,
            posterior_sample,
            keyarray,
            hpdi_values,
            median_value,
            gammaLkeyarray,
            polyfunc1,
            polyfunc2,
            polyfunc3,
            polyfunc4,
        )

    (
        Tcenter1,
        Tcenter2,
        Tcenter3,
        Tcenter4,
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfunc1,
        polyfunc2,
        polyfunc3,
        polyfunc4,
    ) = calc_keyarray(
        T_seal,
        VMR,
        P0_total_1,
        P0_total_2,
        P0_total_3,
        P0_total_4,
        polyx,
        polynomial,
        model_c,
        rng_key_,
        mcmc,
    )

    print_results(
        file1,
        file2,
        file3,
        file4,
        P_total1,
        P_total2,
        P_total3,
        P_total4,
        Tcenter1,
        Tcenter2,
        Tcenter3,
        Tcenter4,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
    )

    # Call the function with appropriate parameters
    # Plot the spectra and fits
    def plot_save(
        wavd1,
        Trans1,
        wavd2,
        Trans2,
        wavd3,
        Trans3,
        wavd4,
        Trans4,
        nu_center_1st,
        mcmc,
        savefilename,
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfunc1,
        polyfunc2,
        polyfunc3,
        polyfunc4,
        yoffset,
    ):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 24))
        plt.rcParams["font.size"] = 14

        # Plot median results
        ax.plot(
            wavd1, median_value["y1"][::-1], color="C0", label="HMC Median Result 1"
        )
        ax.plot(
            wavd2,
            median_value["y2"][::-1] + yoffset,
            color="C2",
            label="HMC Median Result 2",
        )
        ax.plot(
            wavd3,
            median_value["y3"][::-1] + yoffset * 2,
            color="C3",
            label="HMC Median Result 3",
        )
        ax.plot(
            wavd4,
            median_value["y4"][::-1] + yoffset * 3,
            color="C4",
            label="HMC Median Result 4",
        )

        # Plot HPDI (Highest Posterior Density Interval)
        ax.fill_between(
            wavd1,
            hpdi_values["y1"][0][::-1],
            hpdi_values["y1"][1][::-1],
            alpha=0.3,
            interpolate=True,
            color="C0",
            label="90% area 1",
        )
        ax.fill_between(
            wavd2,
            hpdi_values["y2"][0][::-1] + yoffset,
            hpdi_values["y2"][1][::-1] + yoffset,
            alpha=0.3,
            interpolate=True,
            color="C2",
            label="90% area 2",
        )
        ax.fill_between(
            wavd3,
            hpdi_values["y3"][0][::-1] + yoffset * 2,
            hpdi_values["y3"][1][::-1] + yoffset * 2,
            alpha=0.3,
            interpolate=True,
            color="C3",
            label="90% area 3",
        )
        ax.fill_between(
            wavd4,
            hpdi_values["y4"][0][::-1] + yoffset * 3,
            hpdi_values["y4"][1][::-1] + yoffset * 3,
            alpha=0.3,
            interpolate=True,
            color="C4",
            label="90% area 4",
        )

        # Plot measured spectra
        ax.plot(wavd1, Trans1, ".", color="black", label="Measured spectra 1")
        ax.plot(
            wavd2,
            Trans2 + yoffset,
            ".",
            color="black",
            label=f"Measured spectra 2, offset={yoffset:.2g}",
        )
        ax.plot(
            wavd3,
            Trans3 + yoffset * 2,
            ".",
            color="black",
            label=f"Measured spectra 3, offset={yoffset * 2:.2g}",
        )
        ax.plot(
            wavd4,
            Trans4 + yoffset * 3,
            ".",
            color="black",
            label=f"Measured spectra 4, offset={yoffset * 3:.2g}",
        )

        ax.plot(
            wavd1,
            polyfunc1[::-1],
            "--",
            linewidth=2,
            color="C0",
            label="Polynomial Component (median) 1",
        )
        ax.plot(
            wavd2,
            polyfunc2[::-1] + yoffset,
            "--",
            linewidth=2,
            color="C2",
            label="Polynomial Component (median) 2",
        )
        ax.plot(
            wavd3,
            polyfunc3[::-1] + yoffset * 2,
            "--",
            linewidth=2,
            color="C3",
            label="Polynomial Component (median) 3",
        )
        ax.plot(
            wavd4,
            polyfunc4[::-1] + yoffset * 3,
            "--",
            linewidth=2,
            color="C4",
            label="Polynomial Component (median) 4",
        )

        # Plot Voigt fitted line centers
        linecenter1 = 1e7 / (nu_center_1st + median_value["nu_offset1"])
        linecenter2 = 1e7 / (nu_center_1st + median_value["nu_offset2"])
        linecenter3 = 1e7 / (nu_center_1st + median_value["nu_offset3"])
        linecenter4 = 1e7 / (nu_center_1st + median_value["nu_offset4"])
        linemax = 1
        linemin = 0.95

        plt.vlines(
            linecenter1,
            linemin,
            linemax,
            linewidth=2,
            color="C0",
            label="Voigt fitted line center (median) 1",
        )
        plt.vlines(
            linecenter2,
            linemin + yoffset,
            linemax + yoffset,
            linewidth=2,
            color="C2",
            label="Voigt fitted line center (median) 2",
        )
        plt.vlines(
            linecenter3,
            linemin + yoffset * 2,
            linemax + yoffset * 2,
            linewidth=2,
            color="C3",
            label="Voigt fitted line center (median) 3",
        )
        plt.vlines(
            linecenter4,
            linemin + yoffset * 3,
            linemax + yoffset * 3,
            linewidth=2,
            color="C4",
            label="Voigt fitted line center (median) 4",
        )

        # Plot settings
        plt.xlabel("wavelength (nm)")
        plt.ylabel("Intensity Ratio")
        plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
        ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
        ax.minorticks_on()
        plt.tick_params(labelsize=16)
        ax.get_xaxis().get_major_formatter().set_useOffset(
            False
        )  # To avoid exponential labeling
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=10, ncol=2)
        plt.ylim(np.min(Trans4 + yoffset * 3) + yoffset, np.max(Trans1) - yoffset * 0.5)

        # Text annotation for fit parameters
        plt.text(
            0.99,
            1.01,
            "γ0 = {:#.2g} +{:#.2g} -{:#.2g}".format(
                median_value["gamma_self"],
                hpdi_values["gamma_self"][1] - median_value["gamma_self"],
                median_value["gamma_self"] - hpdi_values["gamma_self"][0],
            )
            + ", n = {:#.2g} +{:#.2g} -{:#.2g}".format(
                median_value["n"],
                hpdi_values["n"][1] - median_value["n"],
                median_value["n"] - hpdi_values["n"][0],
            )
            + ", sigma1 = {:#.2g}".format(median_value["sigma1"])
            + ", sigma2 = {:#.2g}".format(median_value["sigma2"])
            + ", sigma3 = {:#.2g}".format(median_value["sigma3"])
            + ", sigma4 = {:#.2g}".format(median_value["sigma4"]),
            fontsize=10,
            va="bottom",
            ha="right",
            transform=ax.transAxes,
        )

        # save the plots
        plt.savefig(savefilename + "_spectra.jpg", bbox_inches="tight")
        # plt.show()
        plt.close()
        print("Spectra plot done!")

        # corner plot
        fontsize = 24
        arviz.rcParams["plot.max_subplots"] = 2000
        arviz.plot_pair(
            arviz.from_numpyro(mcmc),
            # var_names=pararr, #parameter list to display
            kind="kde",
            divergences=True,
            marginals=True,
            colorbar=True,
            textsize=fontsize,
            backend_kwargs={"constrained_layout": True},
            # reference_values=refs,
            # reference_values_kwargs={'color':"red", "marker":"o", "markersize":22},
        )

        plt.savefig(savefilename + "_corner.jpg", bbox_inches="tight", dpi=50)
        # plt.show()
        plt.close()
        print("Corner plot done!")

        # plot the distributions
        arviz.plot_trace(
            mcmc, var_names=keyarray, backend_kwargs={"constrained_layout": True}
        )
        plt.savefig(savefilename + "_plotdist.jpg", bbox_inches="tight")
        # plt.show()
        plt.close()
        print("Distribution plot done!")

        # Output the results to a text file
        with open(f"{savefilename}_results.txt", "w") as f:
            for key in gammaLkeyarray:
                f.write(
                    f"{key},{median_value[key]},{hpdi_values[key][0]},{hpdi_values[key][1]}\n"
                )

        # Save the posterior data
        import pickle

        with open(savefilename + "_post.pkl", "wb") as f:  # "wb"=binary mode(recommend)
            pickle.dump(posterior_sample, f)

    plot_save(
        wavd1,
        Trans1,
        wavd2,
        Trans2,
        wavd3,
        Trans3,
        wavd4,
        Trans4,
        nu_center_1st,
        mcmc,
        savefilename,
        posterior_sample,
        keyarray,
        hpdi_values,
        median_value,
        gammaLkeyarray,
        polyfunc1,
        polyfunc2,
        polyfunc3,
        polyfunc4,
        yoffset,
    )
