# Gascell_Exojax

This repository contains spectral data and modeling scripts for analyzing high-resolution methane absorption spectra in H₂/He gas environments used in Hosokawa et al., 2025. 
The data and code are designed for use with [ExoJAX](https://github.com/HajimeKawahara/exojax), a GPU-accelerated spectral modeling tool for exoplanetary and substellar atmospheres.

## Repository Structure
### 📁 Spectral Data Files
These files provide 8 sets of normalized intensity spectra of methane (CH₄) within the wavelength range 1600–1630 nm, under various temperature and volume mixing ratio (VMR) conditions. The naming convention is:  
1600-1630nm_CH4_VMR[VMR]_T[Temperature]K_IntensityNormalized.dat

For example:
- `1600-1630nm_CH4_VMR01_T1000K_IntensityNormalized.dat`  
  → CH₄ spectra at 1000 K with VMR = 0.1
- `1600-1630nm_CH4_VMR01_T297K_IntensityNormalized.dat`  
  → Room temperature spectrum

Note that only the spectrum of VMR=0.1, 700K has data in the range of 1610-1630nm due to the setting of the measurement.
The intensity is defined as the average intensity over the measured wavelength range set to 1, and is different from the transmittance.

### 🧮 Python Scripts

- `HMC_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha.py`:  
  Performs Voigt profile fitting using a Hamiltonian Monte Carlo (HMC) approach across 8 spectral datasets. It estimates the multiple parameters, including the pressure broadening for CH₄ under mixed gas conditions.

- `HMC_defs_8data_nsep.py`:  
  Contains shared functions and definitions used across HMC modeling scripts, including model setup.

- `Isobaric_Numdensity.py`:  
  Computes number densities under isobaric conditions based on temperature and gas constants.

- `Leastsquare_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha.py`:  
  Uses least-squares optimization instead of Bayesian inference to fit Voigt profiles across the same dataset.

- `Trans_model_MultiVoigt_HITEMP_nu_nsep.py`:  
  Constructs transmission models used in multi-Voigt fitting with input from the HITEMP database.



## Getting Started

This repository assumes that you have ExoJAX installed and are familiar with its API for forward modeling and retrievals. If not, please see the [ExoJAX](https://github.com/HajimeKawahara/exojax) documentation for setup instructions.


### Usage

To fit parameters using Bayesian inference:

python HMC_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha.py

To run least-square optimization instead:

python Leastsquare_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha.py

You can modify temperature, VMR, and wavelength ranges by editing the input filenames and parameters in the script headers.



