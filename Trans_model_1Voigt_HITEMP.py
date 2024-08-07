# calculate the Transmission model with Voigt×1 + cross-section
# updated 2024/08/06


# Import the modules
from exojax.utils.constants import Patm, Tref_original  # [bar/atm]
from exojax.spec.hitran import line_strength, doppler_sigma, gamma_hitran
from exojax.spec.premodit import xsmatrix_zeroth
from exojax.spec import initspec, voigt
from exojax.spec.lpf import xsmatrix as lpf_xsmatrix
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)


# using opa.xsmatrix, lpf.xsmatrix
# wavenumber order
def Trans_model_1Voigt(
    nu_offset,
    alpha,
    gamma_self,
    n,
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
    # include the offset to wavenumber grid
    nu_data_offset_grid = nu_data + nu_offset

    # create the pressure array
    P_total_array = np.full(len(Tarr), P_total)
    P_self_array = np.full(len(Tarr), P_self)

    # cross-section matrix of weak lines at each Temperature channel
    xsmatrix_weak = opa.xsmatrix(Tarr, P_total_array)

    # Calculation for Voigt fitting
    # doppler width
    doppler_1st = jit(vmap(doppler_sigma, (None, 0, None)))(
        mdb_voigt.nu_lines, Tarr, mdb_voigt.molmass
    )
    # partition function
    qt = vmap(mdb_voigt.qr_interp_lines)(Tarr)

    # line strength
    SijM = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        Tarr,
        mdb_voigt.logsij0,
        mdb_voigt.nu_lines,
        mdb_voigt.elower,
        qt,
        mdb_voigt.Tref,
    )

    # Lorentz width
    gamma_L_1st = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))(
        P_total_array,
        Tarr,
        P_self_array,
        n,
        np.zeros_like(mdb_voigt.gamma_self),
        gamma_self,
    )

    # create wavenumber matrix
    nu_matrix = initspec.init_lpf(mdb_voigt.nu_lines, nu_grid)

    # cross section
    # xsmatrix_target = alpha * lpf_xsmatrix(nu_matrix, doppler_1st, gamma_L_1st, SijM)
    # xsmatrix_all = xsmatrix_target + xsmatrix_weak
    xsmatrix_target = lpf_xsmatrix(nu_matrix, doppler_1st, gamma_L_1st, SijM)
    xsmatrix_all = alpha * (xsmatrix_target + xsmatrix_weak)

    tau_length = nMolecule[:, np.newaxis] * xsmatrix_all * Lbin
    tau_length_alllayer = tau_length.sum(axis=0)

    # transmittance
    trans_all = jnp.exp(-tau_length_alllayer)

    # downsampling along to the instrumental resolution, include the effect of offset, trim the adjust range region
    trans_all_specgrid = sop_inst.sampling(trans_all, 0, nu_data_offset_grid)

    return trans_all_specgrid


# ---------------------------------------------------------------------------------
