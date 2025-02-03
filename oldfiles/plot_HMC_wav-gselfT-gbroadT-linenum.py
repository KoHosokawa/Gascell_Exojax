import matplotlib.pyplot as plt
import numpy as np
from exojax.utils.constants import Patm, Tref_original  # [bar/atm]
import scienceplots
from exojax.spec.hitran import gamma_hitran
import os

plt.style.use(["science", "nature"])

# Constants
T0 = 273.15  # celsius to kelvin
VMR = 1
xoffset = 0.01
specdirec = "Results/HMC/Multifit/"

# Inputs
Tcenter_list_model1 = [300, 500, 700, 1000]
Tcen_ind = 3


def calculate_tratio(tcenter_list):
    """Calculate the temperature ratio with respect to Tref_original."""
    return [T / Tref_original for T in tcenter_list]


def load_data_from_file(file, param_name):
    """Load and parse data from the specified file, dynamically defining endings."""
    data = {
        suffix: {
            "values": [],
            "lower_limits": [],
            "upper_limits": [],
            "values_ratio": [],
            "lower_limits_ratio": [],
            "upper_limits_ratio": [],
        }
        for suffix in ["300", "500", "700", "1000"]
    }

    # Dynamically track the endings found in the file
    endings = set()

    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if parts[0].startswith(param_name):
                # Split the parameter name by underscores
                params = parts[0].split("_")

                # Extract the second last and last elements
                suffix = params[-2]  # e.g., "300"
                ending = "_" + params[-1]  # e.g., "_1"

                # Track unique endings found in the file
                endings.add(ending)

                if suffix in data:
                    value = float(parts[1])
                    lower_limit = value - float(parts[2])
                    upper_limit = float(parts[3]) - value

                    data[suffix]["values"].append(value)
                    data[suffix]["lower_limits"].append(lower_limit)
                    data[suffix]["upper_limits"].append(upper_limit)
                    data[suffix]["values_ratio"].append(1.0)
                    data[suffix]["lower_limits_ratio"].append(lower_limit / value)
                    data[suffix]["upper_limits_ratio"].append(upper_limit / value)

    # Convert the set of endings to a sorted list
    endings = sorted(endings)

    return data, endings


def plot_HMC_data(file, param_name, lambda_list, color, label, ax, label_drawn_flags):
    """Plot model data from the specified file."""
    data, endings = load_data_from_file(file, param_name)  # Unpack the tuple
    suffixes = ["300", "500", "700", "1000"]

    # Determine the number of endings dynamically based on the length of values list
    sample_suffix = suffixes[
        0
    ]  # Use the first suffix to get the length of the values list
    num_endings = len(data[sample_suffix]["values"])  # Determine the number of endings

    suffix = suffixes[Tcen_ind]
    if data[suffix]["values"]:
        values = data[suffix]["values"]
        lower_limits = data[suffix]["lower_limits"]
        upper_limits = data[suffix]["upper_limits"]

        for j in range(num_endings):
            ending = endings[j]

            # Plot error bars, only draw label once
            ax.errorbar(
                lambda_list[j],
                values[j],
                yerr=[[lower_limits[j]], [upper_limits[j]]],
                markersize=7,
                elinewidth=3,
                capsize=7,
                fmt="o",
                color=color,
                alpha=0.9,
                label=(
                    label if not label_drawn_flags[label] else None
                ),  # Check if label was drawn
            )
            label_drawn_flags[label] = True  # Mark label as drawn


def read_hitran_data(file):
    """Read HITRAN data from the specified file and return relevant lists."""
    lambda_list, line_numbers, gamma_self_list, gamma_air_list, n_air_list = (
        [],
        [],
        [],
        [],
        [],
    )

    with open(file, "r") as f:
        for line in f:
            if line.startswith("Voigt Line center"):
                parts = line.strip().split(",")
                line_number = int(parts[0].split()[3])
                line_numbers.append(line_number)

                for part in parts:
                    if "λ=" in part:
                        # Extract the value of lambda and remove the 'nm' unit
                        lambda_value = float(
                            part.split("=")[1].replace("nm", "").strip()
                        )
                        lambda_list.append(lambda_value)
                    elif "gamma_self=" in part:
                        gamma_self_list.append(float(part.split("=")[1]))
                    elif "gamma_air=" in part:
                        gamma_air_list.append(float(part.split("=")[1]))
                    elif "n_air=" in part:
                        n_air_list.append(float(part.split("=")[1]))

    return lambda_list, line_numbers, gamma_self_list, gamma_air_list, n_air_list


def plot_HMC_data(
    file, param_name, index_list, color, label, ax, label_drawn_flags, marker
):
    """Plot model data from the specified file, using the provided index_list for X-axis."""
    data, endings = load_data_from_file(file, param_name)  # Unpack the tuple
    suffixes = ["300", "500", "700", "1000"]

    # Determine the number of endings dynamically based on the length of values list
    sample_suffix = suffixes[
        0
    ]  # Use the first suffix to get the length of the values list
    num_endings = len(data[sample_suffix]["values"])  # Determine the number of endings

    suffix = suffixes[Tcen_ind]
    if data[suffix]["values"]:
        values = data[suffix]["values"]
        lower_limits = data[suffix]["lower_limits"]
        upper_limits = data[suffix]["upper_limits"]

        for j in range(num_endings):
            ending = endings[j]

            # Plot error bars, only draw label once, using index_list[j] for X-axis
            ax.errorbar(
                index_list[j],  # X-axis is now index_list[j]
                values[j],
                yerr=[[lower_limits[j]], [upper_limits[j]]],
                markersize=7,
                elinewidth=3,
                capsize=7,
                fmt=marker,
                color=color,
                alpha=0.9,
                label=(
                    label if not label_drawn_flags[label] else None
                ),  # Check if label was drawn
            )
            label_drawn_flags[label] = True  # Mark label as drawn


def plot_hitran_data(
    Tcenter_list_model,
    index_list,
    line_numbers,
    gamma_self_list,
    gamma_air_list,
    n_air_list,
    VMR,
    marker,
    label,
    ax,
    label_drawn_flags,
):
    """Plot HITRAN data using the provided index_list instead of wavelength for X-axis."""
    gammaL_HITEMP_list = []
    for j in range(len(line_numbers)):
        P_total = 1.0 * Patm  # Assuming Patm is defined somewhere
        P_self = P_total * VMR  # Assuming VMR is defined somewhere

        T_center = Tcenter_list_model[Tcen_ind]
        gammaL_HITEMP = gamma_hitran(
            P_total,
            T_center,
            P_self,
            n_air_list[j],
            gamma_air_list[j],
            gamma_self_list[j],
        )
        gammaL_HITEMP_list.append(gammaL_HITEMP)

    ax.scatter(
        index_list,  # X-axis is now the provided index_list
        gammaL_HITEMP_list,
        marker=marker,
        s=70,  # Size of the markers, adjust as needed
        color="black",
        label=(
            label if not label_drawn_flags[label] else None
        ),  # Check if label was drawn
    )
    label_drawn_flags[label] = True  # Mark label as drawn


def process_inputs(input_files, param_name, ax):
    """Process multiple input files and plot the data with shared indices for the same element."""
    label_drawn_flags = {
        "gamma_self": False,
        "gamma_H2+He": False,
        "HITEMP gamma-self": False,
        "HITEMP gamma-air": False,
    }

    start_index = 0  # Initialize starting index for continuous X-axis

    for input_file in input_files:
        # Read HITRAN data
        lambda_list, line_numbers, gamma_self_list, gamma_air_list, n_air_list = (
            read_hitran_data(input_file)
        )

        # Create an index list for this file, starting from the back
        index_list = list(
            range(start_index + len(lambda_list) - 1, start_index - 1, -1)
        )

        # Plot HMC data, using the same index_list for both plots
        plot_HMC_data(
            input_file,
            param_name[0],
            index_list,
            "C1",
            "gamma_self",
            ax,
            label_drawn_flags,
            "o",
        )
        plot_HMC_data(
            input_file,
            param_name[1],
            index_list,
            "C2",
            "gamma_H2+He",
            ax,
            label_drawn_flags,
            "s",
        )

        # Plot HITRAN data, using the same index_list for both plots
        plot_hitran_data(
            Tcenter_list_model1,
            index_list,
            line_numbers,
            gamma_self_list,
            gamma_air_list,
            n_air_list,
            1,
            "o",
            "HITEMP gamma-self",
            ax,
            label_drawn_flags,
        )
        plot_hitran_data(
            Tcenter_list_model1,
            index_list,
            line_numbers,
            gamma_self_list,
            gamma_air_list,
            n_air_list,
            0,
            "x",
            "HITEMP gamma-air",
            ax,
            label_drawn_flags,
        )

        # Update the start_index for the next file
        start_index += len(lambda_list)


# Example usage
input_files = [
    "240701-0703_161365-1613735_Res00025_CH4VMR01-1_HMC3000_2Voigt-multi_sig1e+3_expdist_nuoff-01_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_161502-161515_Res00025_CH4VMR01-1_HMC3000_2Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_161569-161584_Res00025_CH4VMR01-1_HMC3000_3Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_161767-161777_Res00025_CH4VMR01-1_HMC3000_2Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_16196-161985_R00025_CH4VMR01-1_HMC3000_2V_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_16218-162189_Res00025_CH4VMR01-1_HMC3000_3Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_162319-162334_Res00025_CH4VMR01-1_HMC3000_2Voigt-multi_sig1e+3_expdist_nuoff-01_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_162395-16242_Res00025_CH4VMR01-1_HMC3000_3Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_162559-162569_Res00025_CH4VMR01-1_HMC3000_3Voigt-multi_sig1e+3_expdist_nuoff-01_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_162611-16262_Res00025_CH4VMR01-1_HMC3000_1Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-0703_162713-162724_Res00025_CH4VMR01-1_HMC3000_2Voigt-multi_sig1e+3_expdist_nuoff-005_n0-2_gself-gbroad0-02_pval3_al0-3_unidist_8dfit_results.txt",
    "240701-03_162818-162830_R00025_CH4VMR01-1_HMC3000_3V_sig1e3_expdist_nuoff005_n0-2_gboth0-02_pval3_al0-3_unidist_8dfit_results.txt",
    # Add more input files if needed
]

# Automatically prepend specdirec to all input files
input_files = [os.path.join(specdirec, file) for file in input_files]

# Plot configuration
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
plt.rcParams.update({"font.size": 20})

# Process the input files
process_inputs(input_files, ["gself_Tatm", "gbroad_Tatm"], ax1)

# Label and save the plot
# Reference spectra plot stays the same
ax1.set_xlabel("Line number(from shorter wavelength)", fontsize=20)
ax1.set_ylabel("$\gamma(T)/atm$", fontsize=20)
ax1.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
ax1.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
ax1.tick_params(labelsize=18)
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
plt.title(
    f"$\gamma_{{self}}, \gamma_{{H2+He}}(T={Tcenter_list_model1[Tcen_ind]}K, P=1atm)$ at each line, error range=90\%",
    fontsize=20,
)
ax1.get_xaxis().get_major_formatter().set_useOffset(
    False
)  # To avoid exponential labeling

ax1.set_ylim(0, 0.10)


# plt.savefig("test.jpg", bbox_inches="tight")

plt.savefig(
    specdirec
    + f"8dfit_gself-gbroad-T{Tcenter_list_model1[Tcen_ind]}K-peratm_plot_linenum2.jpg",
    # bbox_inches="tight",
)


plt.show()
plt.close()
print("linenum-plot done!")
