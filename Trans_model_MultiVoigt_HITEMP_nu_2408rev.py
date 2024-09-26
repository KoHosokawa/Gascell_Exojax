# calculate the Transmission model with Voigt×1 + cross-section
# updated 2024/08/06


# Import the modules
from exojax.utils.constants import Patm, Tref_original  # [bar/atm]
from exojax.spec.hitran import line_strength, doppler_sigma, gamma_hitran
from exojax.spec.lpf import xsmatrix as lpf_xsmatrix
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from jax.config import config
from exojax.spec import initspec, voigt


config.update("jax_enable_x64", True)


# multiply the alpha at Sij
def Trans_model_MultiVoigt(
    nu_offset,
    alphas,
    gamma_refs,
    gamma_selfs,
    ns,
    Tarr,
    P_total,
    P_self,
    L,
    nMolecule,
    nu_grid,
    nu_data,
    mdb_voigt,  # masked mdb for Voigt fitting
    opa,
    sop_inst,
):

    Lbin = L / len(Tarr)  # spliting the bins
    # include the offset to wavenumber grid(it is shifted to the oppsite direction of actual offset)
    nu_data_offset_grid = nu_data - nu_offset
    # nu_data_offset_grid = nu_data

    # create the pressure array
    P_total_array = np.full(len(Tarr), P_total)
    P_self_array = np.full(len(Tarr), P_self)

    # cross-section matrix of weak lines at each Temperature channel
    xsmatrix_weak = opa.xsmatrix(Tarr, P_total_array)

    # Calculation for Voigt fitting
    # doppler width
    doppler_array = jit(vmap(doppler_sigma, (None, 0, None)))(
        mdb_voigt.nu_lines, Tarr, mdb_voigt.molmass
    )
    # partition function
    Tref_array = np.full(len(Tarr), Tref_original)
    qt = vmap(mdb_voigt.qr_interp_lines)(Tarr, Tref_array)

    # line strength
    SijM = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        Tarr,
        mdb_voigt.logsij0,
        mdb_voigt.nu_lines,
        mdb_voigt.elower,
        qt,
        Tref_original,
    )

    # Lorentz width
    gamma_L_voigt = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))(
        P_total_array,
        Tarr,
        P_self_array,
        ns,
        gamma_refs,
        gamma_selfs,
    )

    # create wavenumber matrix
    nu_matrix = initspec.init_lpf(mdb_voigt.nu_lines, nu_grid)
    # nu_matrix = initspec.init_lpf(mdb_voigt.nu_lines + nu_offset, nu_grid) #if you want to separate the nu_offset
    # print("shapes: ", nu_matrix.shape, doppler_1st.shape, gamma_L_1st.shape, SijM.shape)

    # cross section
    xsmatrix_target = lpf_xsmatrix(
        nu_matrix, doppler_array, gamma_L_voigt, SijM * alphas
    )
    xsmatrix_all = xsmatrix_target + xsmatrix_weak

    tau_length = nMolecule[:, np.newaxis] * xsmatrix_all * Lbin
    tau_length_alllayer = tau_length.sum(axis=0)

    # transmittance
    trans_all = jnp.exp(-tau_length_alllayer)

    # downsampling along to the instrumental resolution, include the effect of offset, trim the adjust range region
    trans_all_specgrid = sop_inst.sampling(trans_all, 0, nu_data_offset_grid)

    return trans_all_specgrid


def Trans_model_MultiVoigt_part(
    nu_offset,
    alphas,
    gamma_refs,
    gamma_selfs,
    ns,
    Tarr,
    P_total,
    P_self,
    L,
    nMolecule,
    nu_grid,
    nu_data,
    mdb_voigt,  # masked mdb for Voigt fitting
    opa,
    sop_inst,
):

    Lbin = L / len(Tarr)  # spliting the bins
    # include the offset to wavenumber grid(it is shifted to the oppsite direction of actual offset)
    nu_data_offset_grid = nu_data - nu_offset

    # create the pressure array
    P_total_array = np.full(len(Tarr), P_total)
    P_self_array = np.full(len(Tarr), P_self)

    # cross-section matrix of weak lines at each Temperature channel
    xsmatrix_weak = opa.xsmatrix(Tarr, P_total_array)

    # Calculation for Voigt fitting
    # doppler width
    doppler_array = jit(vmap(doppler_sigma, (None, 0, None)))(
        mdb_voigt.nu_lines, Tarr, mdb_voigt.molmass
    )
    # partition function
    Tref_array = np.full(len(Tarr), Tref_original)
    qt = vmap(mdb_voigt.qr_interp_lines)(Tarr, Tref_array)

    # line strength
    SijM = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        Tarr,
        mdb_voigt.logsij0,
        mdb_voigt.nu_lines,
        mdb_voigt.elower,
        qt,
        Tref_original,
    )

    # Lorentz width
    gamma_L_voigt = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))(
        P_total_array,
        Tarr,
        P_self_array,
        ns,
        gamma_refs,
        gamma_selfs,
    )

    # create wavenumber matrix
    nu_matrix = initspec.init_lpf(mdb_voigt.nu_lines, nu_grid)
    # print("shapes: ", nu_matrix.shape, doppler_1st.shape, gamma_L_1st.shape, SijM.shape)

    # cross section
    xsmatrix_target = lpf_xsmatrix(
        nu_matrix, doppler_array, gamma_L_voigt, SijM * alphas
    )
    xsmatrix_all = xsmatrix_target + xsmatrix_weak

    tau_length = nMolecule[:, np.newaxis] * xsmatrix_all * Lbin
    tau_length_alllayer = tau_length.sum(axis=0)

    # transmittance
    trans_all = jnp.exp(-tau_length_alllayer)

    # downsampling along to the instrumental resolution, include the effect of offset, trim the adjust range region
    trans_all_specgrid = sop_inst.sampling(trans_all, 0, nu_data_offset_grid)

    tau_length_target = nMolecule[:, np.newaxis] * xsmatrix_target * Lbin
    tau_length_weak = nMolecule[:, np.newaxis] * xsmatrix_weak * Lbin
    tau_length_target_alllayer = tau_length_target.sum(axis=0)
    tau_length_weak_alllayer = tau_length_weak.sum(axis=0)

    # transmittance
    trans_target = jnp.exp(-tau_length_target_alllayer)
    trans_weak = jnp.exp(-tau_length_weak_alllayer)

    # downsampling along to the instrumental resolution, include the effect of offset, trim the adjust range region
    trans_all_specgrid = sop_inst.sampling(trans_all, 0, nu_data_offset_grid)
    trans_target_specgrid = sop_inst.sampling(trans_target, 0, nu_data_offset_grid)
    trans_weak_specgrid = sop_inst.sampling(trans_weak, 0, nu_data_offset_grid)

    return trans_target_specgrid, trans_weak_specgrid, trans_all_specgrid


