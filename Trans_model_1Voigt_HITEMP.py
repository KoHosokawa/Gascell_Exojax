#calculate the Transmission model with Voigt×1 + cross-section 
#updated 2024/07/21
#Import the modules
from exojax.utils.constants import Patm, Tref_original #[bar/atm]
from exojax.spec.hitran import line_strength, doppler_sigma, gamma_hitran
from exojax.spec.premodit import xsmatrix_zeroth
from exojax.spec import initspec,voigt
from exojax.spec.opacalc import OpaPremodit
import jax.numpy as jnp
from jax import vmap
import numpy as np
import tqdm
from jax.config import config
config.update("jax_enable_x64", True)
from IPython.display import clear_output
import matplotlib.pyplot as plt

#using opapremodit.xsmatrix
def Trans_model_1Voigt_opa(Wavoff, S_0,alpha, gamma_self, n, Tarr,Twt, P_total,P_self, L, nMolecule,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst): 
    #Create the transmission data array
    trans =jnp.ones(len(nu_data_adjust)) #initialaization of Transmittance
    trans_weak =jnp.ones(len(nu_data_adjust))
    
    Lbin = L/len(Tarr)  #split the path length in temperature channels
    wavnm = wav/10 #For converting AA to nm (ascending order)
    nu_center_1st = mdb.nu_lines[strline_ind]   #line center wavenumber of strongest line
    nu_offset  = Wavoff * 1E+7/(wavnm[0] *(wavnm[0] -Wavoff)) #convert the offset in wavelength to wavenumber
    nu_data_offset_grid = nu_data_adjust + nu_offset    #include the offset to wavenumber grid
    
    Wavoff_weak = Wavoff #assuming the  weaklines have same wavelength offsets to strongest line
    nu_weak_offset  = Wavoff_weak * 1E+7/(wavnm[0] *(wavnm[0] -Wavoff_weak)) 
    nu_weak_offset_grid = nu_data_adjust + nu_weak_offset

    
    #Create the mdb_weak that removed the strengest line
    nu_lines_weak = np.delete(mdb.nu_lines, strline_ind)
    elower_weak = np.delete(mdb.elower, strline_ind) 
    gamma0_weak = np.delete(mdb.gamma_air/Patm, strline_ind)#[cm-1/atm]to[cm-1/bar]
    n_weak = np.delete(mdb.n_air, strline_ind)
    line_strength_ref_weak = np.delete(mdb.line_strength_ref, strline_ind)

    mdb_weak = mdb
    mdb_weak.nu_lines = nu_lines_weak
    mdb_weak.elower = elower_weak 
    mdb_weak.gamma_air = gamma0_weak
    mdb_weak.n_air = n_weak
    mdb_weak.line_strength_ref = line_strength_ref_weak
    #clear_output() #delete the above outputs. you can't see the run process
    
    '''
    #create the grids for calculating dE at opapremodit (without strongest line)
        lbd, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = initspec.init_premodit(
            nu_lines_weak, #read the parameter value from the file in mdbCO_orig directry for PreMODIT
            nu_grid,
            elower_weak,
            gamma0_weak,#change to h2 if you model the H2-atmosphere 
            n_weak,
            line_strength_ref_weak,
            Twt=Twt, #temperature for weight(K)
            Tref=Tref_original,#reference temperature of premodit grid
            Tref_broadening =Tref_original,#reference temperature for broadening
            warning=False)
    '''
    opa = OpaPremodit(mdb=mdb_weak,
                      nu_grid=nu_grid,
                      #diffmode=0, #i-th Taylor expansion is used for the weight, default is 0.
                      auto_trange=(np.min(Tarr),np.max(Tarr))
                      #dit_grid_resolution=0.2, #It ignores tha value of broadening_resolution
                      #manual_params=[elower_grid[1]-elower_grid[0],Tref,Twt], #dE~160
                      #broadening_resolution={"mode": "single", "value":[gamma, n_Texp]},
                      # #broadening_resolution={'mode': 'manual', 'value': 1.0},#mode: 'single' or 'manual' or 'minmax'. 
                      )#opacity calculation
        
        
    #cross-section of weak lines at each Temperature channel
    P_total_array = np.full(len(Tarr),P_total)
    xsmatrix_weak = opa.xsmatrix(Tarr, P_total_array) 
    
    #Calculate the Voigt fitting with separating several channels
    for j in range(len(Tarr)):
        Tarr_j = Tarr[j]
        nMolecule_j = nMolecule[j]
        beta_1st = doppler_sigma(nu_center_1st,Tarr_j,mdb.molmass)
        S_T_1st = line_strength(Tarr_j, jnp.log(S_0), nu_center_1st, mdb.elower, mdb.qr_interp_lines(Tarr_j), Tref_original)[strline_ind]
        gamma_L_1st = gamma_hitran(P_total,Tarr_j,P_self,n,0,gamma_self ) #P[bar],gamma_self[cm-1/atm],gamma_L_1st[cm-1/atm]
        trans *= jnp.exp(-alpha * S_T_1st * nMolecule_j * voigt(nu_data_offset_grid-nu_center_1st, beta_1st, gamma_L_1st) * Lbin)
        
        #downsampling and fix the wavenumber grid alng to the evenly spaced grid in wavelength
        xsmatrix_weak_specgrid_offset = sop_inst.sampling(xsmatrix_weak[j], 0, nu_weak_offset_grid)
        #calculate the Transmittance of each temperature channel
        trans_weak *= jnp.exp(-nMolecule[j]* xsmatrix_weak_specgrid_offset * Lbin)
        
    #calculate the Transmittance
    trans_all = trans_weak * trans 
    trans_wav = trans_all[::-1] #Invert alng to wavelength ascending order
    trans_trim = trans_wav[start_idx:start_idx + slicesize] #cut ou the adjust range
    return trans_trim
#---------------------------------------------------------------------------------




#using xsmatrix_zeroth
def Trans_model_1Voigt(Wavoff, S_0,alpha, gamma_self, n, Tarr,Twt, P_total,P_self, L, nMolecule,strline_ind,nu_grid,nu_data_adjust,wav,mdb, start_idx, slicesize,sop_inst): 
    #Create the transmission data array
    trans =jnp.ones(len(nu_data_adjust)) #initialaization of Transmittance
    trans_weak =jnp.ones(len(nu_data_adjust))
    
    Lbin = L/len(Tarr)  #split the path length in temperature channels
    wavnm = wav/10 #For converting AA to nm (ascending order)
    nu_center_1st = mdb.nu_lines[strline_ind]   #line center wavenumber of strongest line
    nu_offset  = Wavoff * 1E+7/(wavnm[0] *(wavnm[0] -Wavoff)) #convert the offset in wavelength to wavenumber
    nu_data_offset_grid = nu_data_adjust + nu_offset    #include the offset to wavenumber grid
    
    Wavoff_weak = Wavoff #assuming the  weaklines have same wavelength offsets to strongest line
    nu_weak_offset  = Wavoff_weak * 1E+7/(wavnm[0] *(wavnm[0] -Wavoff_weak)) 
    nu_weak_offset_grid = nu_data_adjust + nu_weak_offset
    
    #Calculate the Transmittance with separating several channels
    for j in range(len(Tarr)):
        Tarr_j = Tarr[j]
        nMolecule_j = nMolecule[j]
        beta_1st = doppler_sigma(nu_center_1st,Tarr_j,mdb.molmass)
        S_T_1st = line_strength(Tarr_j, jnp.log(S_0), nu_center_1st, mdb.elower, mdb.qr_interp_lines(Tarr_j), Tref_original)[strline_ind]
        gamma_L_1st = gamma_hitran(P_total,Tarr_j,P_self,n,0,gamma_self ) #P[bar],gamma_self[cm-1/atm],gamma_L_1st[cm-1/atm]
        
        #Create the mdb parameters that removed the strengest line
        nu_lines_weak = np.delete(mdb.nu_lines, strline_ind)
        elower_weak = np.delete(mdb.elower, strline_ind) 
        gamma0_weak = np.delete(mdb.gamma_air/Patm, strline_ind)#[cm-1/atm]to[cm-1/bar]
        n_weak = np.delete(mdb.n_air, strline_ind)
        line_strength_ref_weak = np.delete(mdb.line_strength_ref, strline_ind)
        qt = vmap(mdb.qr_interp, (None, 0))(mdb.isotope, jnp.array([Tarr_j]))
        #clear_output() #delete the above outputs. you can't see the run process
        print(len(mdb.nu_lines))

        #create the grids for calculating dE at opapremodit (without strongest line)
        lbd, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = initspec.init_premodit(
            nu_lines_weak, #read the parameter value from the file in mdbCO_orig directry for PreMODIT
            nu_grid,
            elower_weak,
            gamma0_weak,#change to h2 if you model the H2-atmosphere 
            n_weak,
            line_strength_ref_weak,
            Twt=Twt, #temperature for weight(K)
            Tref=Tref_original,#reference temperature of premodit grid
            Tref_broadening =Tref_original,#reference temperature for broadening
            warning=False)
        
        #cross-section of weak lines
        xsmatrix_weak = xsmatrix_zeroth(jnp.array([Tarr_j]), jnp.array([P_total]) ,Tref_original, R, pmarray,
                                   lbd, nu_grid, ngamma_ref_grid,
                                   n_Texp_grid, multi_index_uniqgrid,
                                   elower_grid, mdb.molmass, qt,
                                   Tref_original)[0]
        

        #downsampling and fix the wavenumber grid alng to the evenly spaced grid in wavelength
        xsmatrix_weak_specgrid_offset = sop_inst.sampling(xsmatrix_weak, 0, nu_weak_offset_grid)

        #calculate the Transmittance of each temperature channel
        trans_weak *= jnp.exp(-nMolecule_j * xsmatrix_weak_specgrid_offset * Lbin)
        trans *= jnp.exp(-alpha * S_T_1st * nMolecule_j * voigt(nu_data_offset_grid-nu_center_1st, beta_1st, gamma_L_1st) * Lbin)
        
    #calculate the Transmittance
    trans_all = trans_weak * trans 
    trans_wav = trans_all[::-1] #Invert alng to wavelength ascending order
    trans_trim = trans_wav[start_idx:start_idx + slicesize] #cut ou the adjust range
    return trans_trim
    
    
#---------------------------------------------------------------------------------
