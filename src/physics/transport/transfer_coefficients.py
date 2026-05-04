"""
@module physics.transport.transfer_coefficients
@brief Mass and heat transfer coefficients for 1D packed-bed columns.

@details
Two computation modes:

    "constant"  — coefficients taken directly from transport config
                  (already tiled to (nc, N) or (N,) in build_transport_config).
                  No dimensionless numbers are computed.

    "correlation" — coefficients computed from semi-empirical correlations:

        Mass transfer (adsorption equipment):
            Sh = 2 + 1.1 · Re^0.6 · Sc^(1/3)    → k_mtc
            D_ax = u · d_p / Pe_particle           → D_disp  (Bodenstein)

        Heat transfer (all packed-bed equipment):
            Nu_bed  = 2 + 1.1 · Re^0.6 · Pr^(1/3)  → h_bed   (Ranz-Marshall)
            Nu_wall = 3.66 (laminar) or
                      0.023 Re^0.8 Pr^0.4 (turbulent, Re_crit=2300) → h_wall

        Film properties for bed coefficients evaluated at Tfilm = 0.5*(Tg + Ts).

Axial dispersion regime classification (requires L parameter):

    Pe_bed = u · L / D_ax

    Pe_bed > Pe_plugflow  →  "convection"  (D_ax zeroed out — plug flow)
    Pe_disponly ≤ Pe_bed ≤ Pe_plugflow  →  "full"
    Pe_bed < Pe_disponly  →  "dispersion"  (D_ax kept; convection negligible)

    This is analogous to the Re < 2300 / Re ≥ 2300 threshold for h_wall.
    Pe thresholds are read from trans_config (keys Pe_plugflow, Pe_disponly).
    When L is None the classification is skipped (backward-compatible).

Optional key handling:
    k_mtc / D_disp / k_mtc_fixed / D_disp_fixed are absent for equipment
    without adsorption (e.g. gasifier). All accesses use .get() so the
    function works correctly for both adsorber and gasifier trans_config dicts.

Units: SI.
  k_mtc  [1/s]    — LDF overall mass-transfer coefficient
  D_disp [m²/s]   — axial dispersion per component and cell
  h_bed  [W/m²/K] — gas-solid HTC
  h_wall [W/m²/K] — gas-wall HTC
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.pure_gas import eval_species_property
from src.physics.thermodynamics.mixture import wilke_mix_property
from src.physics.transport.nusselt import h_wall_tube, compute_Ra_Di
from src.utils.profiling import profiled


@profiled
def compute_transfer_coefficients(
    Tg: np.ndarray,
    Ts: np.ndarray,
    x: np.ndarray,
    gas_props: dict,
    u_rel: np.ndarray,
    prop_gas: dict,
    prop_lecho: dict,
    Di: float,
    trans_config: dict,
    n_comp: int,
    N: int,
    Tw: np.ndarray | None = None,
    L: float | None = None,
) -> dict:
    """
    Compute mass and heat transfer coefficients for the RHS of a packed-bed column.

    Parameters
    ----------
    Tg        : ndarray(N,)    gas temperature per cell [K]
    Ts        : ndarray(N,)    solid temperature per cell [K]
    x         : ndarray(N, nc) molar fraction per cell and component [-]
    gas_props : dict           mixture properties (from compute_gas_mixture_properties)
    u_rel     : ndarray(N,)    relative velocity |v_cell| [m/s]
    prop_gas  : dict           pure-gas properties (for correlation mode)
    prop_lecho: dict           bed properties (D_p, a_surf per cell)
    Di        : float          column inner diameter [m]
    trans_config: dict         result of build_transport_config
    n_comp    : int            number of components
    N         : int            number of cells
    Tw        : ndarray(N,) or None  wall temperature [K]; activates Ra (natural
                               convection) in h_wall when provided.
    L         : float or None  bed length [m]; when provided, enables Pe_bed
                               regime classification for axial dispersion.
                               None → skip classification (backward-compatible).

    Returns
    -------
    dict with mandatory keys:
        k_mtc     : ndarray(nc, N) or None  [1/s]    — None for non-adsorption equipment
        D_disp    : ndarray(nc, N) or None  [m²/s]   — None when plug-flow
        h_bed     : ndarray(N,)             [W/m²/K]
        h_wall    : ndarray(N,)             [W/m²/K]
    and optional diagnostic keys (correlation mode only):
        Pr, Re, Sc, Sh, Nu_bed, Nu_wall, Tfilm, mu_film, k_film, Cp_molar_film
        Pe_bed    : ndarray(N,) or None  [—]  axial Péclet at bed scale
        regime_disp : ndarray(N,) or None  str  "convection"|"full"|"dispersion"
    """
    nc, nn = int(n_comp), int(N)
    mode = trans_config["mode"]

    Tg_arr    = np.asarray(Tg,    dtype=float).reshape(-1)
    Ts_arr    = np.asarray(Ts,    dtype=float).reshape(-1)
    x_mat     = np.asarray(x,     dtype=float).reshape(nn, nc)
    u_rel_arr = np.asarray(u_rel, dtype=float).reshape(-1)

    if mode == "constant":
        # k_mtc / D_disp are absent in reactors without adsorption (e.g. gasifier)
        return {
            "k_mtc":      trans_config.get("k_mtc"),
            "D_disp":     trans_config.get("D_disp"),
            "h_bed":      trans_config["h_bed"],
            "h_wall":     trans_config["h_wall"],
            "Pe_bed":     None,
            "regime_disp": None,
        }

    # =========================================================
    # "correlation" mode
    # =========================================================
    MW_arr    = np.asarray(prop_gas["MW"],     dtype=float).reshape(-1)
    MWmix_arr = np.asarray(gas_props["MWmix"], dtype=float).reshape(-1)
    rho_arr   = np.asarray(gas_props["rho"],   dtype=float).reshape(-1)
    Dim_mat   = np.asarray(gas_props["Dim"],   dtype=float).reshape(nn, nc)

    d_p_arr    = np.asarray(prop_lecho["D_p"],    dtype=float).reshape(-1)
    a_surf_arr = np.asarray(prop_lecho["a_surf"], dtype=float).reshape(-1)
    Di_val     = float(Di)

    # Film properties at Tfilm = 0.5*(Tg + Ts)
    Tfilm_arr = 0.5 * (Tg_arr + Ts_arr)

    mu_i_film  = eval_species_property(prop_gas["mu"],       Tfilm_arr, nc)  # (N, nc)
    k_i_film   = eval_species_property(prop_gas["k"],        Tfilm_arr, nc)
    Cp_i_film  = eval_species_property(prop_gas["Cp_molar"], Tfilm_arr, nc)

    mu_film_arr       = wilke_mix_property(x=x_mat, prop_i=mu_i_film, MW=MW_arr)
    k_film_arr        = wilke_mix_property(x=x_mat, prop_i=k_i_film,  MW=MW_arr)
    Cp_molar_film_arr = np.sum(x_mat * Cp_i_film, axis=1)
    Cp_mass_film_arr  = Cp_molar_film_arr / np.maximum(MWmix_arr, 1.0e-30)

    # Dimensionless numbers
    Pr_arr  = Cp_mass_film_arr * mu_film_arr / np.maximum(k_film_arr, 1.0e-30)
    Sc_mat  = mu_film_arr[:, None] / np.maximum(rho_arr[:, None] * Dim_mat, 1.0e-30)
    Re_arr  = rho_arr * u_rel_arr * d_p_arr / np.maximum(mu_film_arr, 1.0e-30)

    # ── Mass transfer (k_mtc) ────────────────────────────────────────────────
    Sh_mat      = 2.0 + 1.1 * (Re_arr[:, None] ** 0.6) * (Sc_mat ** (1.0 / 3.0))
    beta_mt_mat = Sh_mat * Dim_mat / d_p_arr[:, None]
    k_mtc_mat   = (beta_mt_mat * a_surf_arr[:, None]).T                     # (nc, N)

    # ── Axial dispersion (D_disp) — Bodenstein + Pe_bed classification ───────
    _D_disp_fixed = trans_config.get("D_disp_fixed")
    Pe_part       = float(trans_config.get("Pe_particle", 2.0))
    Pe_cf         = float(trans_config.get("Pe_plugflow", 100.0))
    Pe_cd         = float(trans_config.get("Pe_disponly",  0.1))

    if _D_disp_fixed is None:
        # Bodenstein correlation: D_ax = u · d_p / Pe_particle
        D_ax_ref = u_rel_arr * d_p_arr / max(Pe_part, 1.0e-30)             # (N,) [m²/s]
        D_disp_mat = np.tile(D_ax_ref[None, :], (nc, 1))                    # (nc, N)
    else:
        _fix = np.asarray(_D_disp_fixed, dtype=float)
        if _fix.ndim == 0:
            # Scalar override (gasifier): same value for all species
            D_ax_ref   = float(_fix) * np.ones(nn, dtype=float)             # (N,)
            D_disp_mat = np.tile(D_ax_ref[None, :], (nc, 1))                # (nc, N)
        else:
            # Per-species array (adsorber): (nc,) → (nc, N)
            _fix_arr   = _fix.reshape(-1)
            D_ax_ref   = float(np.mean(_fix_arr)) * np.ones(nn, dtype=float)
            D_disp_mat = np.tile(_fix_arr[:, None], (1, nn))                # (nc, N)

    # Pe_bed regime classification — only when bed length L is provided
    if L is not None:
        Pe_bed_arr  = u_rel_arr * float(L) / np.maximum(D_ax_ref, 1.0e-300) # (N,)
        # Plug-flow cells: zero out D_disp (convection term sufficient)
        plug_mask   = Pe_bed_arr > Pe_cf                                     # (N,) bool
        D_disp_mat[:, plug_mask] = 0.0
        # Regime label per cell (diagnostic)
        regime_arr = np.where(
            Pe_bed_arr > Pe_cf, "convection",
            np.where(Pe_bed_arr < Pe_cd, "dispersion", "full"),
        )
    else:
        Pe_bed_arr = None
        regime_arr = None

    # ── Bed heat transfer (h_bed) — Ranz-Marshall ────────────────────────────
    Nu_bed_arr = 2.0 + 1.1 * (Re_arr ** 0.6) * (Pr_arr ** (1.0 / 3.0))
    h_bed_arr  = Nu_bed_arr * k_film_arr / d_p_arr

    # ── Wall heat transfer (h_wall) — Dittus-Boelter / Nu=3.66 + Ra ─────────
    Re_wall_arr = rho_arr * u_rel_arr * Di_val / np.maximum(mu_film_arr, 1.0e-30)
    if Tw is not None and trans_config.get("h_wall_fixed") is None:
        Ra_wall_arr = compute_Ra_Di(
            Tg=Tg_arr, Tw=np.asarray(Tw, dtype=float).reshape(-1),
            rho_Tg=rho_arr, mu_film=mu_film_arr,
            k_film=k_film_arr, Cp_film=Cp_mass_film_arr,
            Di=Di_val,
        )
    else:
        Ra_wall_arr = None
    h_wall_arr  = h_wall_tube(Re=Re_wall_arr, Pr=Pr_arr, D=Di_val,
                               k_film=k_film_arr, Ra=Ra_wall_arr)
    Nu_wall_arr = h_wall_arr * Di_val / np.maximum(k_film_arr, 1.0e-30)

    # ── Override with fixed values if supplied ───────────────────────────────
    _k_mtc_fixed  = trans_config.get("k_mtc_fixed")
    _h_bed_fixed  = trans_config.get("h_bed_fixed")
    _h_wall_fixed = trans_config.get("h_wall_fixed")
    if _k_mtc_fixed is not None:
        k_mtc_mat = np.tile(_k_mtc_fixed.reshape(nc, 1), (1, nn))
    if _h_bed_fixed is not None:
        h_bed_arr = _h_bed_fixed * np.ones(nn, dtype=float)
    if _h_wall_fixed is not None:
        h_wall_arr = _h_wall_fixed * np.ones(nn, dtype=float)

    return {
        "k_mtc":         k_mtc_mat,
        "D_disp":        D_disp_mat,
        "h_bed":         h_bed_arr,
        "h_wall":        h_wall_arr,
        "Pe_bed":        Pe_bed_arr,
        "regime_disp":   regime_arr,
        "Pr":            Pr_arr,
        "Sc":            Sc_mat,
        "Re":            Re_arr,
        "Sh":            Sh_mat,
        "beta_mt":       beta_mt_mat,
        "Nu_bed":        Nu_bed_arr,
        "Nu_wall":       Nu_wall_arr,
        "Tfilm":         Tfilm_arr,
        "mu_film":       mu_film_arr,
        "k_film":        k_film_arr,
        "Cp_molar_film": Cp_molar_film_arr,
    }
