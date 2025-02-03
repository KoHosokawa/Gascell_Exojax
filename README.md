This repository stores the codes for the analysis of Gascell experiment.  
"HMC_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha-Vcom-Tcom.py": runs the Bayesian inference by HMC sampling.
"Leastsquare_MultiVoigt_8dfit_g-H2He_gself_nsep_alpha-Vcom-Tcom.py": runs the estimation of the parameters by Least-square methods discribed in Append. B of the paper.

Those files use the modules descibed in below files:
"Isobaric_Numdensity.py": returns the total pressure and number density of each Temperature points.  
"Trans_model_MultiVoigt_HITEMP_nu_nsep.py": returns the Transmittance calculated from HITEMP database.  
"HMC_defs_8data_nsep.py": is the definition file to output the graphs, corner plots, etc.  


