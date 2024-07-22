# %% [markdown]
# This script runs HMC including Teperature Gradients & weak lines for gamma self
# 2024/07/21 Updated(removed alpha from weak cross-section,removed add_data, sepalated the file about Transmission calculation)
# 2024/07/15 Updated(sop.sample for wavgrid, nu_offset correction. introduced gridboost)

# %%
##Read (& Show) the spectra data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

#Road the spectra file
direc1="../../実験/Data/240701/Corrected/"
direc2="../../実験/Data/240701/Corrected/"
direc3="../../実験/Data/240701/Corrected/"
direc4="../../実験/Data/240703/Corrected/"

file1 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_300K_wmask_1103_1_AveNorm_fringe_removed.dat"
file2 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_500K_wmask_45dMt-X0V-Y75V_1322_1_AveNorm_fringe_removed.dat"
file3 = "240701_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_700K_wmask_45dMt-X75V-Y0V_1515_1_AveNorm_fringe_removed.dat"
file4 = "240703_wav1600-1630_res00025_0dbm_5kVW_ch1-1path-CH4VMR1-P04W-rot30d_ch2-ref_SRQopt_1000K_wmask_45dMt-150V-Y150V_1416_1_AveNorm_fringe_removed.dat"

Data1 = np.genfromtxt(direc1 + file1, delimiter=' ', skip_header=0)
Data2 = np.genfromtxt(direc2 + file2, delimiter=' ', skip_header=0)
Data3 = np.genfromtxt(direc3 + file3, delimiter=' ', skip_header=0)
Data4 = np.genfromtxt(direc4 + file4, delimiter=' ', skip_header=0)

#Trim the data amount
start = 8712
end = 8773
data3_offset = 0
wavd1 = Data1[start:end,0]
Trans1 = Data1[start:end,1]
wavd2 = Data2[start:end,0]
Trans2 = Data2[start:end,1]

wavd3 = Data3[start+data3_offset:end+data3_offset,0]
Trans3 = Data3[start+data3_offset:end+data3_offset,1]
wavd4 = Data4[start:end,0]
Trans4 = Data4[start:end,1]

'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
print("Data points = " + str(len(Trans1))+ ", " +str(len(Trans2)))
print("Wavelength Ranges = ",wavd1[0],"-",wavd1[-1],"nm, ",wavd2[0],"-",wavd2[-1],"nm",wavd3[0],"-",wavd3[-1],"nm, ",wavd4[0],"-",wavd4[-1],"nm")

#plt.plot(wavd,nTrans,'.-',alpha=0.5,color="gray")
plt.plot(wavd1,Trans1,'.-',alpha=1,color="C0",label="Data1")
plt.plot(wavd2,Trans2,'.-',alpha=1,color="C2",label="Data2")
plt.plot(wavd3,Trans3,'.-',alpha=1,color="C1",label="Data3")
plt.plot(wavd4,Trans4,'.-',alpha=1,color="C3",label="Data4")
plt.xlabel("wavelength $\AA$")
plt.legend()
plt.grid(which = "major", axis = "both", alpha = 0.7,linestyle = "--", linewidth = 1)
ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
ax.minorticks_on()
ax.get_xaxis().get_major_formatter().set_useOffset(False) #To avoid exponential labeling


plt.show()
plt.clf() #clear the plot
plt.close() #close the plot window
'''




# %%
# Import the modules for Running HMC
from exojax.utils.constants import Patm, Tref_original,Tc_water #[bar/atm]
from exojax.utils.grids import wavenumber_grid #, grid_resolution
from exojax.spec.api import MdbHitemp
from exojax.spec.hitran import line_strength, gamma_hitran
from exojax.spec.specop import SopInstProfile
from Isobaric_Numdensity import cal_numd_P
from Trans_model_1Voigt_HITEMP import Trans_model_1Voigt, Trans_model_1Voigt_opa
#from exojax.spec.opacalc import OpaPremodit
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.config import config
config.update("jax_enable_x64", True)

import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random
from IPython.display import clear_output

#Constants
#T0 = 273.15 #celsius to kelvin

#parameter Setting
'''
Tarr1=np.array([24.0,24.0,24.0,24.0,23.9,23.9,23.7,23.6])+Tc_water #CH3-10 of 2024/06/27 at 300K
Tarr2=np.array([231.0,230.6,230.2,231.0,230.9,231.3,233.2,233.3])+Tc_water #CH3-10 of 2024/06/27 at 500K
Tarr3=np.array([429.6,428.1,429.5,431.6,431.6,432.0,434.0,432.0])+Tc_water #CH3-10 of 2024/05/27 at 700K
Tarr4=np.array([720.1,722.2,728.1,733.5,732.9,731.0,729.4,725.4])+Tc_water #CH3-10 of 2024/06/28 at 1000K
'''
Tarr1=jnp.array([23.5,23.6,23.4,23.5,23.4,23.4,23.4,23.3])+Tc_water #CH3-10 of 2024/07/01 at 300K
Tarr2=jnp.array([228.8,227.7,230.3,231.5,231.0,231.7,234.2,219.6])+Tc_water #CH3-10 of 2024/07/01 at 500K
Tarr3=jnp.array([426.3,426.4,431.9,433.3,431.7,431.8,434.7,413.5])+Tc_water #CH3-10 of 2024/07/01 at 700K
Tarr4=jnp.array([720.5,720.0,727.2,734.3,733.8,731.4,729.7,725.2])+Tc_water #CH3-10 of 2024/07/03 at 1000K

T_seal = 23.6+Tc_water #[K], The temperature at the time the cell was sealed
Twt  =1000 #Weighting Temperature
Resolution = 0.0025 #Data grid resolutionin wavelength[nm]
VMR=1 #volume mixing ratio
L=49.7 #path length[cm]
P0_total_1 = 0.423 #Pressure at the time the cell was sealed
P0_total_2 = 0.423 
P0_total_3 = 0.423 
P0_total_4 = 0.423 

'''
ngas1 = number_density(P0_total_1,T_seal)#P is [bar], unit = [cm^-3]
ngas2 = number_density(P0_total_2,T_seal)
ngas3 = number_density(P0_total_3,T_seal)
ngas4 = number_density(P0_total_4,T_seal)
'''

#calculate the number density and pressure at given temperatures
ngas1,P_total1 = cal_numd_P(Tarr1,P0_total_1,T_seal)
ngas2,P_total2 = cal_numd_P(Tarr2,P0_total_2,T_seal)
ngas3,P_total3 = cal_numd_P(Tarr3,P0_total_3,T_seal)
ngas4,P_total4 = cal_numd_P(Tarr4,P0_total_4,T_seal)

#pressure of target molecule
P_self1 = P_total1*VMR #[atm]
P_self2 = P_total2*VMR
P_self3 = P_total3*VMR
P_self4 = P_total4*VMR

#Molecular number density array considering VMR (cgs)
nMolecule1=VMR*ngas1 
nMolecule2=VMR*ngas2
nMolecule3=VMR*ngas3
nMolecule4=VMR*ngas4

wspan =wavd1[-1]- wavd1[0]  #wavelength span[nm]
adjustrange = 0.05  #additional wavelength range for calculating the cross-section at the edge of wavelength range  
gridboost = 10  #boosting factor of wavenumber resolution
wav_n = round(wspan/Resolution) +1  #number of data points
valrange = 10   #maximum value factor for the range of the 1st~3rd degree of polynomial
polyx = np.linspace(0, wspan, wav_n)    #wavelength bin for polynomial
wmin = round(wavd1[0],5) 
wmax = round(wavd1[-1],5)
start_idx = round(adjustrange/Resolution)   #cut the shorter wavelength region from the adjusrange-included model
slicesize = round(wspan/Resolution+1)   #calculating by np or jnp may cause integer problem
Nx = round(((wspan + 2 * adjustrange)/Resolution + 1))  #Data points including adjust range
Nx_boost = Nx *gridboost    #boosted datapoints
wavd_adjust = np.linspace(wmin-adjustrange, wmax+adjustrange, Nx)   #wavelength grid including adjust range
nu_data_adjust = 1E+7/wavd_adjust[::-1] #wavenumber grid evenly spaced in wavelength
#print(1E+7/nu_data_adjust)

#Calculate the Line strength S(T)
def S_Tcalc(nu, S_0,T):
    logeS_0 = jnp.log(S_0)
    qr=mdb.qr_interp_lines(T)
    return line_strength(T, logeS_0, nu, mdb.elower, qr, Tref_original)
    #return line_strength(T, logeS_0, nu, mdb.elower[strline_ind], qr, Tref)


#polynominal fit function
def polynomial(a,b,c,d,x):
    return a * x**3 + b * x**2 + c * x + d


#generate the wavenumber&wavelength grid for cross-section
nu_grid, wav, res = wavenumber_grid(wmin - adjustrange, 
                                    wmax + adjustrange,
                                    #jnp.max(wavd),
                                    Nx_boost,
                                    unit="nm",
                                    xsmode="premodit",
                                    wavelength_order='ascending')

sop_inst = SopInstProfile(nu_grid)

#Read the line database 
mdb = MdbHitemp('.database/CH4/',
                nurange=nu_grid,
                #crit =1E-30, 
                gpu_transfer=False,#Trueだと計算速度低下
                with_error=True) #for obtaining the error of each line

print("mdb=",len(mdb.nu_lines))

#Calculate the line index in the order of the line strength at T=twt
S_T = S_Tcalc(jnp.exp(mdb.nu_lines),mdb.logsij0,Twt)
strline_ind = jnp.argsort(S_T)[::-1][0]

#Sampling parameter setting
def model_c(y1,y2,y3,y4):
    #wavelength offset for each spectra
    Wavoff1 = numpyro.sample('Wavoff1', dist.Uniform(-0.05, 0.05))
    Wavoff2 = numpyro.sample('Wavoff2', dist.Uniform(-0.05, 0.05))
    Wavoff3 = numpyro.sample('Wavoff3', dist.Uniform(-0.05, 0.05))
    Wavoff4 = numpyro.sample('Wavoff4', dist.Uniform(-0.05, 0.05))
    
    #Line strength factor which is normalized the S(T) for strongest line 
    alpha_range = 1
    alpha1 = numpyro.sample('alpha1', dist.Uniform(1-alpha_range,1+alpha_range))
    alpha2 = numpyro.sample('alpha2', dist.Uniform(1-alpha_range,1+alpha_range))
    alpha3 = numpyro.sample('alpha3', dist.Uniform(1-alpha_range,1+alpha_range))
    alpha4 = numpyro.sample('alpha4', dist.Uniform(1-alpha_range,1+alpha_range))

    #broadening parameters
    gamma_self = numpyro.sample('gamma_self',dist.Uniform(0.0,1.0)) #[cm-1/atm]
    n = numpyro.sample('n',dist.Uniform(-2.,2.))
    
    #polynomial coefficients
    a1 = numpyro.sample('a1', dist.Uniform(-valrange/wspan**2,valrange/wspan**2))
    b1 = numpyro.sample('b1', dist.Uniform(-valrange/wspan,valrange/wspan))
    c1 = numpyro.sample('c1', dist.Uniform(-valrange,valrange))
    d1 = numpyro.sample('d1', dist.Uniform(0., 2.))
    a2 = numpyro.sample('a2', dist.Uniform(-valrange/wspan**2,valrange/wspan**2))
    b2 = numpyro.sample('b2', dist.Uniform(-valrange/wspan,valrange/wspan))
    c2 = numpyro.sample('c2', dist.Uniform(-valrange,valrange))
    d2 = numpyro.sample('d2', dist.Uniform(0., 2.))
    a3 = numpyro.sample('a3', dist.Uniform(-valrange/wspan**2,valrange/wspan**2))
    b3 = numpyro.sample('b3', dist.Uniform(-valrange/wspan,valrange/wspan))
    c3 = numpyro.sample('c3', dist.Uniform(-valrange,valrange))
    d3 = numpyro.sample('d3', dist.Uniform(0., 2.))
    a4 = numpyro.sample('a4', dist.Uniform(-valrange/wspan**2,valrange/wspan**2))
    b4 = numpyro.sample('b4', dist.Uniform(-valrange/wspan,valrange/wspan))
    c4 = numpyro.sample('c4', dist.Uniform(-valrange,valrange))
    d4 = numpyro.sample('d4', dist.Uniform(0., 2.))

    #gaussian noises of each spectra
    sigmain1 = numpyro.sample('sigma1',dist.Exponential(1.0E+3))
    sigmain2 = numpyro.sample('sigma2',dist.Exponential(1.0E+3))
    sigmain3 = numpyro.sample('sigma3',dist.Exponential(1.0E+3))
    sigmain4 = numpyro.sample('sigma4',dist.Exponential(1.0E+3))
    
    #calculate the polynomial
    polyfunc1 = polynomial(a1,b1,c1,d1,polyx)
    polyfunc2 = polynomial(a2,b2,c2,d2,polyx)
    polyfunc3 = polynomial(a3,b3,c3,d3,polyx)
    polyfunc4 = polynomial(a4,b4,c4,d4,polyx)
    
    #Calculate the Transmittance * polynomial
    mu1 = Trans_model_1Voigt(Wavoff1, mdb.line_strength_ref, alpha1, gamma_self, n, Tarr1, Twt, P_total1, P_self1, L,nMolecule1,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst) * polyfunc1
    mu2 = Trans_model_1Voigt(Wavoff2, mdb.line_strength_ref, alpha2, gamma_self, n, Tarr2, Twt, P_total2, P_self2, L,nMolecule2,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst) * polyfunc2
    mu3 = Trans_model_1Voigt(Wavoff3, mdb.line_strength_ref, alpha3, gamma_self, n, Tarr3, Twt, P_total3, P_self3, L,nMolecule3,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst) * polyfunc3
    mu4 = Trans_model_1Voigt(Wavoff4, mdb.line_strength_ref, alpha4, gamma_self, n, Tarr4, Twt, P_total4, P_self4, L,nMolecule4,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst) * polyfunc4
    
    '''
    mu1 = absmodel(Wavoff1, mdb.line_strength_ref, alpha1, gamma_self, n, Tarr1, P0_total_1, nMolecule1,strline_ind) * polyfunc1
    mu2 = absmodel(Wavoff2, mdb.line_strength_ref, alpha2, gamma_self, n, Tarr2, P0_total_2, nMolecule2,strline_ind) * polyfunc2
    mu3 = absmodel(Wavoff3, mdb.line_strength_ref, alpha3, gamma_self, n, Tarr3, P0_total_3, nMolecule3,strline_ind) * polyfunc3
    mu4 = absmodel(Wavoff4, mdb.line_strength_ref, alpha4, gamma_self, n, Tarr4, P0_total_4, nMolecule4,strline_ind) * polyfunc4
    '''
    
    #sample the Transmittance * polynomial with sigmain
    numpyro.sample('y1', dist.Normal(mu1, sigmain1), obs=y1)#obs =jnparray 
    numpyro.sample('y2', dist.Normal(mu2, sigmain2), obs=y2)#obs =jnparray 
    numpyro.sample('y3', dist.Normal(mu3, sigmain3), obs=y3)
    numpyro.sample('y4', dist.Normal(mu4, sigmain4), obs=y4)
    #it samples the normalized distribution of mu. dist.normal: normal distribution of average = mu,std= sigmain
    #print(mu1.shape)


print("Tref =" ,Tref_original)
print("P1 = ",P_total1, "T1 = ",Tarr1)
print("P2 = ",P_total2, "T2 = ",Tarr2)
print("P3 = ",P_total3, "T3 = ",Tarr3)
print("P4 = ",P_total4, "T4 = ",Tarr4)

print("polinomial value range a = ", -valrange/wspan**2, valrange/wspan**2, "b = " ,valrange/wspan, valrange/wspan, "c = " ,valrange, valrange)
print("Model Resolution: R > " + str(wavd1[0]/Resolution)+',Spectral Data point of model = '+str(Nx))
print("Line centers at HITEMP λ= " +str(1.E+7/mdb.nu_lines)+"nm")
print("Strongest Line center at HITEMP λ= " +str(1.E+7/mdb.nu_lines[strline_ind])+"nm, gamma_self="+str(mdb.gamma_self[strline_ind])+", gamma_air="+str(mdb.gamma_air[strline_ind])+", n_air="+str(mdb.n_air[strline_ind]))
gamma0_weak = np.delete(mdb.gamma_air, strline_ind)
n_weak = np.delete(mdb.n_air, strline_ind)
print("weak lines gamma0 = ",gamma0_weak)
print("weak lines n = ",n_weak)

#Run mcmc
rng_key = random.PRNGKey(0) 
rng_key, rng_key_ = random.split(rng_key) #generate random numbers

num_warmup, num_samples = 10, 5
#num_warmup, num_samples = 100, 50
#num_warmup, num_samples = 300, 500
#num_warmup, num_samples = 500, 1000
#num_warmup, num_samples = 1000, 2000
#num_warmup, num_samples = 4000, 6000
kernel = NUTS(model_c, forward_mode_differentiation=False)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, y1=Trans1, y2=Trans2,y3=Trans3,y4=Trans4) #Run the MCMC samplers and collect samples
#initial = {'Wavoff':0.005,'S':0.04, 'beta':0.03, 'gamma0':0.02, 'a':40., 'b':3., 'c':0.2, 'd':1.1}
#mcmc.run(rng_key_, y=Trans,init_params=initial) #Run the MCMC samplers with initial parameters

mcmc.print_summary()  






# %%

#Temperature setting for calculating gammma(T,P)
Tcenter1 = 300 
Tcenter2 = 500
Tcenter3 = 700
Tcenter4 = 1000

#Save file name 
#savefilename="Results/Model+poly/Multifit/240527-0627-0628_162178-162193_Res00025_CH4VMR00981-H5path_SpecCal0611_HMC1500_sigma1e+3-expdist_n-gamma-uniformdist_tgrad_wline-n-gamma-air_gb10_4dfit"
savefilename="Results/Model+poly/Multifit/TEST_240701-0703_162178-162193_Res00025_CH4VMR1-1path_Norm_Fremoved_HMC150_sigma1e+3-expdist_n-gammaself-uniformdist_tgrad_wline-n-gamma-air_gb10_alfvoigt_4dfit"



#Print the input data
print("input file: \n\t", file1, "\n\t", file2,"\n\t", file3, "\n\t", file4)

'''
#Reference Values
refs={};
refs["a1"]=0;refs["b1"]=0.;refs["c1"]=0.;refs["d1"]=1.0;refs["Wavoff1"]=0.;#refs["alpha1"]=0.;L
refs["a2"]=0;refs["b2"]=0.;refs["c2"]=0.;refs["d2"]=1.0;refs["Wavoff2"]=0.;#refs["alpha2"]=0.;
refs["a3"]=0;refs["b3"]=0.;refs["c3"]=0.;refs["d3"]=1.0;refs["Wavoff3"]=0.;#refs["alpha1"]=0.;
refs["a4"]=0;refs["b4"]=0.;refs["c4"]=0.;refs["d4"]=1.0;refs["Wavoff4"]=0.;
refs["n"]=0.5;refs["gamma0"]=0.05;
refs["alpha1"]=1;refs["alpha2"]=1;refs["alpha3"]=1;refs["alpha4"]=1;
refs["sigma1"]=0.003;refs["sigma2"]=0.003;refs["sigma3"]=0.003;refs["sigma4"]=0.003;
'''

diffmode = 0
posterior_sample = mcmc.get_samples()
keyarray = list(posterior_sample.keys()) 
Ykeyarray = keyarray + ['y1','y2','y3','y4']
print("sampled parameter = " ,keyarray)
pred = Predictive(model_c, posterior_sample, return_sites=Ykeyarray) #create the dictionary for all sample sites
predictions = pred(rng_key_, y1=None,y2=None, y3=None,y4=None)


# Calculate HPDI and median values as "hpdi_~" and "median_~"
hpdi_values = {}
median_value = {}

Y_gammaLkeyarray = keyarray + ['y1','y2','y3','y4','gammaL_1','gammaL_2','gammaL_3','gammaL_4']
gammaLkeyarray = keyarray + ['gammaL_1','gammaL_2','gammaL_3','gammaL_4']

for key in Y_gammaLkeyarray:
    if key == 'gammaL_1':
        P_total_1 = P0_total_1 * Tcenter1/T_seal
        P_self_1 = P_total_1*VMR 
        gammaL_1 = gamma_hitran(P_total_1,Tcenter1,P_self_1,predictions['n'],0,predictions['gamma_self']) 
        #gamma_L[cm-1/atm], P unit is [bar]
        hpdi_values[key] = hpdi(gammaL_1, 0.9)
        median_value[key] = np.median(gammaL_1, axis=0)

    elif key == 'gammaL_2':
        P_total_2 = P0_total_2 * Tcenter2/T_seal
        P_self_2 = P_total_2*VMR 
        gammaL_2 = gamma_hitran(P_total_2,Tcenter2,P_self_2,predictions['n'],0,predictions['gamma_self']) 
        #gamma_L[cm-1/atm], P unit is [bar]
        hpdi_values[key] = hpdi(gammaL_2, 0.9)
        median_value[key] = np.median(gammaL_2, axis=0)

    elif key == 'gammaL_3':
        P_total_3 = P0_total_3 * Tcenter3/T_seal
        P_self_3 = P_total_3*VMR 
        gammaL_3 = gamma_hitran(P_total_3,Tcenter3,P_self_3,predictions['n'],0,predictions['gamma_self']) 
        #gamma_L[cm-1/atm], P unit is [bar]
        hpdi_values[key] = hpdi(gammaL_3, 0.9)
        median_value[key] = np.median(gammaL_3, axis=0)

    elif key == 'gammaL_4':
        P_total_4 = P0_total_4 * Tcenter4/T_seal
        P_self_4 = P_total_4*VMR 
        gammaL_4 = gamma_hitran(P_total_4,Tcenter4,P_self_4,predictions['n'],0,predictions['gamma_self']) 
        #gamma_L[cm-1/atm], P unit is [bar]
        hpdi_values[key] = hpdi(gammaL_4, 0.9)
        median_value[key] = np.median(gammaL_4, axis=0)

    else:
        hpdi_values[key] = hpdi(predictions[key], 0.9) #90% range
        
        if key == 'y1' or key == 'y2' or key == 'y3' or key == 'y4': #avoid y since its only array shape
            median_value[key] = jnp.median(predictions[key], axis=0)
        
        else:
            median_value[key] = np.median(posterior_sample[key])
            exec_command_hpdi = 'hpdi_' + str(key) + '=[' + str(hpdi_values[key][0]) +','+ str(hpdi_values[key][1])+']'
            exec_command_median = 'median_' + str(key) + '=' + str(median_value[key]) 
            exec(exec_command_hpdi)
            exec(exec_command_median) 

#print the results
print("gammaL at T = ", Tcenter1, "K, P = {:#.3g}".format(P_total_1), "bar = ", median_value['gammaL_1'])
print("gammaL at T = ", Tcenter2, "K, P = {:#.3g}".format(P_total_2), "bar = ", median_value['gammaL_2'])
print("gammaL at T = ", Tcenter3, "K, P = {:#.3g}".format(P_total_3), "bar = ", median_value['gammaL_3'])
print("gammaL at T = ", Tcenter4, "K, P = {:#.3g}".format(P_total_4), "bar = ", median_value['gammaL_4'])

for key in gammaLkeyarray:
    #print(str(key)+ "= {:#.3g} +{:#.3g} -{:#.3g}".format(median_value[key],hpdi_values[key][1]-median_value[key],median_value[key]-hpdi_values[key][0]))
    print(str(key)+ "= {:#.5g}, Lower: {:#.5g}, Upper: {:#.5g}".format(median_value[key],hpdi_values[key][0], hpdi_values[key][1]))

# Plot the spectra and fits
yoffset = -0.7
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 24))
plt.rcParams["font.size"] = 14
ax.plot(wavd1[:], median_value['y1'], color='C0',label='HMC Median Result')
ax.plot(wavd2[:], median_value['y2']+yoffset, color='C2',label='HMC Median Result 2')
ax.plot(wavd3[:], median_value['y3']+yoffset*2, color='C3',label='HMC Median Result 3')
ax.plot(wavd4[:], median_value['y4']+yoffset*3, color='C4',label='HMC Median Result 4')
ax.fill_between(wavd1[:], hpdi_values['y1'][0], hpdi_values['y1'][1], alpha=0.3, interpolate=True, color='C0', label='90% area 1')
ax.fill_between(wavd2[:], hpdi_values['y2'][0]+yoffset, hpdi_values['y2'][1]+yoffset, alpha=0.3, interpolate=True, color='C2', label='90% area 2')
ax.fill_between(wavd3[:], hpdi_values['y3'][0]+yoffset*2, hpdi_values['y3'][1]+yoffset*2, alpha=0.3, interpolate=True, color='C3', label='90% area 3')
ax.fill_between(wavd4[:], hpdi_values['y4'][0]+yoffset*3, hpdi_values['y4'][1]+yoffset*3, alpha=0.3, interpolate=True, color='C4', label='90% area 4')
ax.plot(wavd1[:], Trans1, '.', color='black', label='Measured spectra 1')
ax.plot(wavd2[:], Trans2+yoffset, '.', color='black', label='Measured spectra 2, offset={:#.2g}'.format(yoffset))
ax.plot(wavd3[:], Trans3+yoffset*2, '.', color='black', label='Measured spectra 3, offset={:#.2g}'.format(yoffset*2))
ax.plot(wavd4[:], Trans4+yoffset*3, '.',  color='black', label='Measured spectra 4, offset={:#.2g}'.format(yoffset*3))

#plot polynomial line
polyfunc1 = polynomial(median_a1,median_b1,median_c1,median_d1,polyx)
polyfunc2 = polynomial(median_a2,median_b2,median_c2,median_d2,polyx)
polyfunc3 = polynomial(median_a3,median_b3,median_c3,median_d3,polyx)
polyfunc4 = polynomial(median_a4,median_b4,median_c4,median_d4,polyx)
ax.plot(wavd1[:], polyfunc1, '--', linewidth=2,color='C0', label='Polynomial Component(median) 1')
ax.plot(wavd2[:], polyfunc2+yoffset, '--', linewidth=2,color='C2', label='Polynomial Component(median) 2')
ax.plot(wavd3[:], polyfunc3+yoffset*2, '--', linewidth=2,color='C3', label='Polynomial Component(median) 3')
ax.plot(wavd4[:], polyfunc4+yoffset*3, '--', linewidth=2,color='C4', label='Polynomial Component(median) 4')

#plot settings
plt.xlabel('wavelength (nm)')
plt.ylabel('Intensity Ratio')
plt.grid(which = "major", axis = "both", alpha = 0.7,linestyle = "--", linewidth = 1)
ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
ax.minorticks_on()
plt.tick_params(labelsize=16)
ax.get_xaxis().get_major_formatter().set_useOffset(False) #To avoid exponential labeling
ax.legend(loc='lower right', bbox_to_anchor=(1,0),fontsize=10,ncol=2)
plt.ylim(np.min(Trans4+yoffset*3)+yoffset,np.max(Trans1)-yoffset*0.5)
plt.text(0.99, 1.01, "γ0 = {:#.2g} +{:#.2g} -{:#.2g}".format(median_gamma_self, hpdi_gamma_self[1] - median_gamma_self, median_gamma_self - hpdi_gamma_self[0])\
         +", n = {:#.2g} +{:#.2g} -{:#.2g}".format(median_n, hpdi_n[1] - median_n, median_n - hpdi_n[0])\
         +", sigma1 = {:#.2g}".format(median_sigma1)+", sigma2 = {:#.2g}".format(median_sigma2)\
         +", sigma3 = {:#.2g}".format(median_sigma3)+", sigma4 = {:#.2g}".format(median_sigma4)\
        ,fontsize=10,  va='bottom', ha='right', transform=ax.transAxes)

#save the plots
plt.savefig(savefilename + "_spectra.jpg",bbox_inches="tight")
plt.show()
plt.close()


#corner plot
fontsize =24
arviz.rcParams["plot.max_subplots"] = 2000
arviz.plot_pair(arviz.from_numpyro(mcmc),
                #var_names=pararr, #parameter list to display
                kind='kde',
                divergences=True,
                marginals=True,
                colorbar = True,
                textsize=fontsize,
                backend_kwargs={"constrained_layout":True},
                #reference_values=refs,
                #reference_values_kwargs={'color':"red", "marker":"o", "markersize":22},
                )

plt.savefig(savefilename + "_corner.jpg",bbox_inches="tight",dpi=50)
plt.show()
plt.close()


#plot the distributions
arviz.plot_trace(mcmc, var_names=keyarray,backend_kwargs={"constrained_layout":True})
plt.savefig(savefilename + "_plotdist.jpg",bbox_inches="tight")
plt.show()
plt.close()



# Output the results to a text file
with open(f"{savefilename}_results.txt", 'w') as f:
    for key in gammaLkeyarray:
        f.write(f'{key},{median_value[key]},{hpdi_values[key][0]},{hpdi_values[key][1]}\n')

#Save the posterior data
import pickle
with open(savefilename+"_post.pkl","wb") as f: #"wb"=binary mode(recommend)
    pickle.dump(posterior_sample, f)

print("Done!")


# %%
#Check if the CPU or GPU is mainly used
import jax
jax.default_backend()
#jax.local_devices()

