"""
@module postprocessing.gasifier_balances
@brief Molar and energy balance verification for the gasifier model.

@details
All balances use the trapezoidal rule for time integration.
A balance is considered closed when:
    |imbalance| / max(|accumulated|, |in/out|) < 1e-2  (1% tolerance)

Sign conventions:
    - Accumulation: end_value - start_value  (positive = net gain)
    - Flux: positive = entering the control volume
    - Reaction sources: positive = net production inside CV
"""

import numpy as np

from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.physics.thermal.wall_heat_flux import wall_heat_flux


# ─── Molar balance ────────────────────────────────────────────────────────────

def molar_balance(gasifier, params: dict) -> dict:
    """
    Verify global molar conservation: gas phase accumulation = net flux + reactions.

    Note: This balance checks consistency of the numerical integration, NOT
    physical conservation (which is enforced by the ODE equations themselves).

    IMPORTANT — reactive systems: in simulations with solid→gas reactions (drying,
    pyrolysis, char conversion), the gas phase gains moles from the solid. These
    reaction sources are NOT subtracted here, so the reported imbalance_rel will
    be large (~1.0) even for a numerically correct integration. The imbalance value
    equals the moles produced by reactions, not a numerical error.
    Use total_mass_balance() for a closure check that holds in reactive systems.

    Parameters
    ----------
    gasifier : SimpleNamespace  output of build_gasifier_results()
    params   : dict

    Returns
    -------
    dict with keys:
        delta_gas_mol    float [mol/m³_gas]  change in total gas moles per m³_gas
        flux_net_mol     float [mol/m³_gas]  net molar flux (in - out)
        imbalance_mol    float [mol/m³_gas]  |delta - flux|
        imbalance_rel    float [-]           relative imbalance
        per_species      dict {str: float}   imbalance per gas species [mol/m³_gas]
    """
    dz    = float(params["dz"])
    epsi_r = float(params["epsi_r"])

    t  = gasifier._t_results            # (n_t,)
    C  = gasifier._C_results            # (n_t, nc, N)
    vin  = gasifier._v_in_results       # (n_t,)
    vout = gasifier._v_out_results      # (n_t,)
    Cin  = gasifier._C_in_results       # (n_t, nc)  NaN if batch

    # Total moles per m³_bed in gas phase: Σ_i C_i * dz (per cell) * epsi_r
    # Summed over all cells: mol_gas_tot / m²_cross_section
    C_tot_spatial = np.sum(C, axis=1)   # (n_t, N)  mol/m³_gas per cell
    mol_gas = np.sum(C_tot_spatial * dz, axis=1)  # (n_t,)  [mol/m²_section]

    delta_gas = mol_gas[-1] - mol_gas[0]

    # Molar flux at inlet and outlet [mol/m²/s]
    C_tot_in  = np.sum(np.nan_to_num(Cin, nan=0.0), axis=1)   # (n_t,)
    flux_in   = vin * C_tot_in * epsi_r                        # [mol/m²_section/s]
    flux_out  = vout * C_tot_spatial[:, -1] * epsi_r           # [mol/m²_section/s]
    flux_net  = np.trapz(flux_in - flux_out, t)                # [mol/m²_section]

    imbalance = abs(delta_gas - flux_net)
    denom     = max(abs(delta_gas), abs(flux_net), 1.0e-30)

    # Per-species
    per_species = {}
    sp = gasifier._species
    for i, name in enumerate(sp):
        mol_i  = np.sum(C[:, i, :] * dz, axis=1)   # (n_t,) [mol/m²]
        delta_i = mol_i[-1] - mol_i[0]
        flux_in_i  = vin * np.nan_to_num(Cin[:, i], nan=0.0) * epsi_r
        flux_out_i = vout * C[:, i, -1] * epsi_r
        flux_net_i = np.trapz(flux_in_i - flux_out_i, t)
        per_species[name] = float(abs(delta_i - flux_net_i))

    return {
        "delta_gas_mol":  float(delta_gas),
        "flux_net_mol":   float(flux_net),
        "imbalance_mol":  float(imbalance),
        "imbalance_rel":  float(imbalance / denom),
        "per_species":    per_species,
    }


# ─── Solid conversion summary ─────────────────────────────────────────────────

def solid_conversion_summary(gasifier, _params: dict) -> dict:
    """
    Report solid conversion state at start and end of simulation.

    Parameters
    ----------
    gasifier : SimpleNamespace
    params   : dict

    Returns
    -------
    dict with keys (all in kg/m³_bed or fraction):
        rho_biomass_0, rho_biomass_f    initial/final biomass bulk density
        rho_char_0,    rho_char_f       initial/final char bulk density
        rho_moisture_0, rho_moisture_f  initial/final moisture bulk density
        X_biomass                       biomass conversion [-]
        X_moisture                      moisture conversion [-]
        X_char                          char conversion [-]
    """
    rho_s = gasifier._rho_solid_results  # (n_t, 3, N)
    # spatial average over N cells
    rho_avg = np.mean(rho_s, axis=2)     # (n_t, 3)

    rb0, rb_f   = rho_avg[0, 0], rho_avg[-1, 0]
    rc0, rc_f   = rho_avg[0, 1], rho_avg[-1, 1]
    rm0, rm_f   = rho_avg[0, 2], rho_avg[-1, 2]

    X_bio = 1.0 - rb_f / max(rb0, 1.0e-12)
    X_moi = 1.0 - rm_f / max(rm0, 1.0e-12)
    # char conversion relative to initial char; can be negative if pyrolysis adds char
    X_chr = 1.0 - rc_f / max(rc0, 1.0e-12)

    return {
        "rho_biomass_0":  float(rb0),
        "rho_biomass_f":  float(rb_f),
        "rho_char_0":     float(rc0),
        "rho_char_f":     float(rc_f),
        "rho_moisture_0": float(rm0),
        "rho_moisture_f": float(rm_f),
        "X_biomass":      float(np.clip(X_bio, 0.0, 1.0)),
        "X_moisture":     float(np.clip(X_moi, 0.0, 1.0)),
        "X_char":         float(X_chr),
    }


# ─── Energy balance ───────────────────────────────────────────────────────────

def energy_balance(gasifier, params: dict) -> dict:
    """
    Verify global energy conservation for gas + solid system.

    Balance equation (integrated over space [0, L] and time [0, t_max]):

        ΔHg + ΔHs ≈ F_h_net + Q_wall_total + Q_rxn_total

    Since reaction heats (Q_rxn) are not stored separately during integration,
    the residual = ΔHg + ΔHs - F_h_net - Q_wall_total ≈ Q_rxn_total.
    For a non-reactive case (drying only) the residual should be near zero.

    Hg convention: Hg [J/m³_bed] = epsi_r * Σ_i C_i * h_i(Tg) where h_i
    are absolute molar enthalpies (including h_formation from gasdb).
    The convective flux uses the same absolute enthalpies, so chemical potential
    changes from reactions are automatically captured through concentration shifts.

    Parameters
    ----------
    gasifier : SimpleNamespace  output of build_gasifier_results()
    params   : dict

    Returns
    -------
    dict with keys [all quantities in J/m²_cross_section unless noted]:
        delta_Hg           change in stored gas volumetric enthalpy
        delta_Hs_approx    change in solid thermal content (Cp*T proxy, approx)
        delta_total        delta_Hg + delta_Hs_approx
        flux_enthalpy_net  convective enthalpy flux net (in − out), integrated
        Q_wall_total       wall heat input to gas, integrated (+ = gas heated)
        h_wall_approx      True if h_wall was not taken from cache/constant config
        residual           delta_total − flux_enthalpy_net − Q_wall_total  ≈ Q_rxn
        residual_rel       residual / |delta_Hg|  (relative to gas enthalpy change)
        note               explanation of residual interpretation
    """
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    epsi_r    = float(params["epsi_r"])
    Ai        = float(params["Ai"])
    Pi        = float(params["Pi"])
    Po        = float(params["Po"])
    gas_T_ref = float(params["gas_T_ref"])
    prop_gas  = params["prop_gas"]
    thermal_bc_config = params["thermal_bc_config"]
    trans_config      = params["trans_config"]
    shell_tube        = params.get("wall_config") is not None

    t    = gasifier._t_results          # (n_t,)
    Hg   = gasifier._Hg_results         # (n_t, N)  [J/m³_bed]
    Tg   = gasifier._Tg_results         # (n_t, N)  [K]
    Ts   = gasifier._Ts_results         # (n_t, N)  [K]
    vin  = gasifier._v_in_results       # (n_t,)    [m/s]
    vout = gasifier._v_out_results      # (n_t,)    [m/s]
    Cin  = gasifier._C_in_results       # (n_t, nc) [mol/m³_gas], NaN if batch
    Tin  = gasifier._T_in_results       # (n_t,)    [K], NaN if batch
    Tw   = gasifier._Tw_results         # (n_t, N) or None

    # ── 1. ΔHg  [J/m²_section] ───────────────────────────────────────────────
    Hg_integral = np.sum(Hg * dz, axis=1)           # (n_t,) [J/m²]
    delta_Hg    = float(Hg_integral[-1] - Hg_integral[0])

    # ── 2. ΔHs approx  [J/m²_section] ───────────────────────────────────────
    # Proxy: Hs ≈ (Σ_j rho_s_j * Cp_s_j(Ts)) * Ts  per cell, summed over cells
    # Uses cell-average Ts (spatial mean) for Cp evaluation — approximate.
    rho_s  = gasifier._rho_solid_results  # (n_t, 3, N)
    Cp_fns = params["solid_config"]["Cp_fns"]
    Ts_bar = np.mean(Ts, axis=1)          # (n_t,) spatial mean [K]
    Cp_bio = np.array([float(Cp_fns[0](T)) for T in Ts_bar])
    Cp_chr = np.array([float(Cp_fns[1](T)) for T in Ts_bar])
    Cp_moi = np.array([float(Cp_fns[2](T)) for T in Ts_bar])
    rho_s_avg = np.mean(rho_s, axis=2)   # (n_t, 3)
    Hs_arr    = (( rho_s_avg[:, 0] * Cp_bio
                 + rho_s_avg[:, 1] * Cp_chr
                 + rho_s_avg[:, 2] * Cp_moi) * Ts_bar * dz * nn)  # (n_t,) [J/m²]
    delta_Hs  = float(Hs_arr[-1] - Hs_arr[0])

    # ── 3. Net convective enthalpy flux  [J/m²_section] ──────────────────────
    # F_h_in  = v_in  * Hg_in  where Hg_in  = epsi_r * Σ_i C_i_in  * h_i(T_in)
    # F_h_out = v_out * Hg_out where Hg_out = _Hg_results[:, -1]   (last cell)
    # Both already include epsi_r factor consistently with the ODE formulation.

    batch_mask = np.isnan(Tin)              # True for batch timesteps
    T_in_safe  = np.where(batch_mask, gas_T_ref, Tin)   # (n_t,) no NaN
    C_in_safe  = np.nan_to_num(Cin, nan=0.0)             # (n_t, nc)

    # Vectorised h_i(T_in) — pass T_in_safe as "spatial" axis: (nc, n_t)
    h_i_in   = calc_species_enthalpy(T_in_safe, prop_gas, nc, gas_T_ref)  # (nc, n_t)
    Hg_in    = epsi_r * np.einsum('it,ti->t', h_i_in, C_in_safe)          # (n_t,) [J/m³_bed]
    Hg_in[batch_mask] = 0.0                # no inlet in batch mode

    F_h_in  = vin  * Hg_in          # (n_t,) [W/m²]
    F_h_out = vout * Hg[:, -1]      # (n_t,) [W/m²]
    F_h_net = float(np.trapz(F_h_in - F_h_out, t))  # [J/m²]

    # ── 4. Q_wall integrated  [J/m²_section] ─────────────────────────────────
    # Determine representative h_wall for modes that need it.
    h_wall_approx = False
    h_wall_val    = None
    cache = params.get("_cache", {})
    if "trans_props" in cache and cache["trans_props"] is not None:
        h_wall_val = float(np.mean(cache["trans_props"]["h_wall"]))
    elif trans_config.get("mode") == "constant":
        h_wall_val = float(trans_config["h_wall"])
    else:
        h_wall_approx = True
        h_wall_val    = 25.0   # nominal [W/m²/K]; flagged in output

    h_wall_arr = np.full(nn, h_wall_val, dtype=float)

    mode = thermal_bc_config["mode"]

    if shell_tube:
        # Gas–inner-wall exchange: qwall_vol = h_wall * Pi/Ai * (Tw - Tg)  [W/m³_bed]
        # Total power to gas: Qwall_dot = h_wall * Pi * dz * Σ_j (Tw_j - Tg_j)  [W]
        if Tw is None:
            Q_wall_total = 0.0
        else:
            dT_gw       = Tw - Tg                                         # (n_t, N) [K]
            Qwall_dot   = h_wall_val * Pi * dz * np.sum(dT_gw, axis=1)    # (n_t,) [W]
            Q_wall_total = float(np.trapz(Qwall_dot / Ai, t))             # [J/m²]
    elif mode == "adiabatic":
        Q_wall_total = 0.0
    elif mode == "heatfluxwall":
        Qwall_fixed  = float(thermal_bc_config["Qwall"])                   # [W]
        Q_wall_total = float(Qwall_fixed * (t[-1] - t[0]) / Ai)           # [J/m²]
    else:
        # "fixed_twall" or "ambient_htc": integrate wall_heat_flux over time
        Qwall_dot = np.zeros(len(t), dtype=float)
        for k in range(len(t)):
            _, Qk, _ = wall_heat_flux(
                Tg=Tg[k], h_wall=h_wall_arr,
                thermal_bc_config=thermal_bc_config,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )
            Qwall_dot[k] = Qk
        Q_wall_total = float(np.trapz(Qwall_dot / Ai, t))  # [J/m²]

    # ── 5. Residual ───────────────────────────────────────────────────────────
    delta_total = delta_Hg + delta_Hs
    residual    = delta_total - F_h_net - Q_wall_total
    denom       = max(abs(delta_Hg), 1.0e-30)

    return {
        "delta_Hg":          float(delta_Hg),
        "delta_Hs_approx":   float(delta_Hs),
        "delta_total":       float(delta_total),
        "flux_enthalpy_net": float(F_h_net),
        "Q_wall_total":      float(Q_wall_total),
        "h_wall_approx":     bool(h_wall_approx),
        "residual":          float(residual),
        "residual_rel":      float(residual / denom),
        "note": (
            "residual = delta_total - flux_net - Q_wall ≈ Q_rxn (reaction heat "
            "released inside reactor). Near zero for non-reactive runs; "
            "ΔHs uses Cp(Ts_mean)*Ts proxy — approximate for large ΔTs."
        ),
    }


# ─── Total mass balance ──────────────────────────────────────────────────────

def total_mass_balance(gasifier, params: dict) -> dict:
    """
    Verify global mass conservation: Δ(mass_gas) + Δ(mass_solid) = net_mass_flux.

    This balance holds for reactive systems because mass is conserved between
    phases: solid → gas transfers decrease mass_solid and increase mass_gas by
    the same amount. The balance reduces to near zero for a correct integration.

    For batch mode (no flow): net_mass_flux = 0, so the check is simply
        Δ(mass_gas) + Δ(mass_solid) ≈ 0

    Unlike molar_balance, this function does NOT report a spurious large imbalance
    when solid→gas reactions are active.

    Parameters
    ----------
    gasifier : SimpleNamespace  output of build_gasifier_results()
    params   : dict

    Returns
    -------
    dict with keys [all in kg/m²_cross_section]:
        delta_mass_gas    change in stored gas mass
        delta_mass_solid  change in stored solid mass
        delta_mass_total  delta_mass_gas + delta_mass_solid
        net_mass_flux     mass entering − mass leaving via convection (integrated)
        imbalance_kg      |delta_mass_total − net_mass_flux|
        imbalance_rel     imbalance_kg / max(|delta_mass_gas|, |delta_mass_solid|)
        note              interpretation guide
    """
    dz     = float(params["dz"])
    epsi_r = float(params["epsi_r"])
    MW     = np.asarray(params["MW"], dtype=float)   # (nc,) [kg/mol]

    t    = gasifier._t_results          # (n_t,)
    C    = gasifier._C_results          # (n_t, nc, N)  [mol/m³_gas]
    vin  = gasifier._v_in_results       # (n_t,)
    vout = gasifier._v_out_results      # (n_t,)
    Cin  = gasifier._C_in_results       # (n_t, nc)  NaN if batch
    rho_s = gasifier._rho_solid_results # (n_t, 3, N) [kg/m³_bed]

    # ── 1. Δ(masa del gas) [kg/m²] ───────────────────────────────────────────
    # mass_gas_cell = epsi_r * Σ_i C_i * MW_i  [kg/m³_bed]
    # Integrated over length: Σ_cell mass_gas_cell * dz
    mass_g_per_vol = epsi_r * np.einsum('tij,i->tj', C, MW)  # (n_t, N) [kg/m³_bed]
    mass_gas = np.sum(mass_g_per_vol * dz, axis=1)            # (n_t,)   [kg/m²]
    delta_mass_gas = float(mass_gas[-1] - mass_gas[0])

    # ── 2. Δ(masa del sólido) [kg/m²] ────────────────────────────────────────
    # mass_solid_cell = Σ_j rho_s_j  [kg/m³_bed]
    mass_s_per_vol = np.sum(rho_s, axis=1)                    # (n_t, N) [kg/m³_bed]
    mass_solid = np.sum(mass_s_per_vol * dz, axis=1)          # (n_t,)   [kg/m²]
    delta_mass_solid = float(mass_solid[-1] - mass_solid[0])

    # ── 3. Flujo másico neto en contornos [kg/m²] ────────────────────────────
    # F_mass_in  = v_in  * (Σ_i C_in_i  * MW_i) * epsi_r   [kg/m²/s]
    # F_mass_out = v_out * (Σ_i C_out_i * MW_i) * epsi_r   [kg/m²/s]
    C_in_safe   = np.nan_to_num(Cin, nan=0.0)                 # (n_t, nc)
    rho_g_in    = np.einsum('ti,i->t', C_in_safe, MW)         # (n_t,) [kg/m³_gas]
    F_mass_in   = vin * rho_g_in * epsi_r                     # (n_t,) [kg/m²/s]

    C_out       = C[:, :, -1]                                  # (n_t, nc)
    rho_g_out   = np.einsum('ti,i->t', C_out, MW)             # (n_t,)
    F_mass_out  = vout * rho_g_out * epsi_r                   # (n_t,)

    net_mass_flux = float(np.trapz(F_mass_in - F_mass_out, t))

    # ── 4. Cierre ─────────────────────────────────────────────────────────────
    delta_mass_total = delta_mass_gas + delta_mass_solid
    imbalance        = abs(delta_mass_total - net_mass_flux)
    denom            = max(abs(delta_mass_gas), abs(delta_mass_solid), 1.0e-30)

    return {
        "delta_mass_gas":   float(delta_mass_gas),
        "delta_mass_solid": float(delta_mass_solid),
        "delta_mass_total": float(delta_mass_total),
        "net_mass_flux":    float(net_mass_flux),
        "imbalance_kg":     float(imbalance),
        "imbalance_rel":    float(imbalance / denom),
        "note": (
            "imbalance = |Δmass_gas + Δmass_solid − flux_net|. "
            "Near zero for a correct integration in any mode (batch/cstr/1D). "
            "Δmass_solid < 0 (solid converts to gas); Δmass_gas > 0; "
            "sum should equal net convective mass flux."
        ),
    }


# ─── Check balances ───────────────────────────────────────────────────────────

def check_balances(gasifier, params: dict, verbose: bool = True) -> dict:
    """
    Informe de balance de conservación para todos los subsistemas del gasificador.

    Formato de salida:
        - Tabla principal: Acumulación | Flujo neto BC | Residual | Res [%]
          Filas: masa total (★ cierre numérico), especies gas, masa sólida
        - Desglose energético: todos los términos del balance de Hg y Ts
        - Veredicto: error máximo del cierre numérico

    Integración: trapezoidal sobre [0,L]×[0,T]. Unidades: [kg/m²], [mol/m²], [J/m²].

    Parameters
    ----------
    gasifier : SimpleNamespace  — output de build_gasifier_results()
    params   : dict
    verbose  : bool             — True imprime el informe

    Returns
    -------
    dict con claves: mass_total, species_gas, solid, energy_gas, energy_solid, energy_wall
    """
    # ── Parámetros ─────────────────────────────────────────────────────────────
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    epsi_r    = float(params["epsi_r"])
    Ai        = float(params["Ai"])
    Pi        = float(params["Pi"])
    Po        = float(params["Po"])
    gas_T_ref = float(params["gas_T_ref"])
    MW        = np.asarray(params["MW"], dtype=float)
    prop_gas  = params["prop_gas"]
    trans_config   = params["trans_config"]
    thermal_bc_cfg = params["thermal_bc_config"]
    shell_tube     = params.get("wall_config") is not None
    species        = list(params["species"])
    h_fns          = params["solid_config"]["h_fns"]    # H_j(T) = ∫_{273}^T Cp_j dT

    # ── Resultados ─────────────────────────────────────────────────────────────
    t     = gasifier._t_results
    Hg    = gasifier._Hg_results                          # (n_t, N) [J/m³_bed]
    Tg    = gasifier._Tg_results                          # (n_t, N) [K]
    Ts    = gasifier._Ts_results                          # (n_t, N) [K]
    Tw    = gasifier._Tw_results                          # (n_t, N) or None
    C     = gasifier._C_results                           # (n_t, nc, N)
    rho_s = gasifier._rho_solid_results                   # (n_t, 3, N)
    vin   = gasifier._v_in_results
    vout  = gasifier._v_out_results
    Cin   = gasifier._C_in_results                        # (n_t, nc) NaN if batch
    Tin   = gasifier._T_in_results

    C_in_safe  = np.nan_to_num(Cin, nan=0.0)
    batch_mask = np.isnan(Tin)

    # Etiqueta descriptiva del punto de operación (solo para display)
    _bcc        = params["bc_config"]
    _v_gas_in   = _bcc.get("v_gas_in")
    _v_solid    = float(_bcc.get("v_solid", 0.0))
    _v_out      = _bcc.get("v_out")       # None | 0.0 | float > 0
    _inlet_mode = str(_bcc.get("inlet_mode", "prescribed"))
    if _v_gas_in is None and (_v_out is None or float(_v_out) == 0.0):
        bc_mode = "batch"
    elif _v_gas_in is None and _v_out is not None and float(_v_out) > 0.0:
        bc_mode = "semibatch"
    elif abs(_v_solid) < 1e-12:
        bc_mode = "cstr"
    elif _inlet_mode == "computed":
        bc_mode = "conveyor"
    elif (str(_bcc.get("direction") or "updraft")) == "updraft":
        bc_mode = "updraft"
    else:
        bc_mode = "downdraft"

    L          = dz * nn

    # Factor de normalización: integrar sobre volumen y tiempo → [J/m³_bed]
    # Divide todos los resultados en [J/m²] por L para obtener [J/m³_bed].
    _norm = 1.0 / L   # [1/m]

    # ── Helpers de formato ────────────────────────────────────────────────────
    W  = 84   # ancho total
    C1 = 22   # ancho columna Variable
    C2 = 14   # ancho columnas numéricas

    def _hdr():
        return (f"  {'Variable':<{C1}}  {'Acumulación':>{C2}}  "
                f"{'Flujo neto BC':>{C2}}  {'Residual':>{C2}}  {'Res [%]':>8}")

    def _trow(label, acum, flux, resid, pct=None, note=""):
        pct_s = f"{pct:>+8.2f}" if pct is not None else f"{'—':>8}"
        n = f"  {note}" if note else ""
        return (f"  {label:<{C1}}  {acum:>{C2}.4e}  "
                f"{flux:>{C2}.4e}  {resid:>{C2}.4e}  {pct_s}{n}")

    def _sep(title="", char="─"):
        if title:
            pad = max(W - len(title) - 5, 2)
            return f"\n  {char*3} {title} {char*pad}"
        return "  " + char * W

    def _ok(rel, thr=0.01):
        return "✓ OK" if abs(rel) < thr else "⚠ REVISAR"

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 1 — MASA  (resultados en [J/m²] internamente, se normalizan al final)
    # ══════════════════════════════════════════════════════════════════════════
    mass_g_vol = epsi_r * np.einsum('tij,i->tj', C, MW)
    mass_gas   = np.sum(mass_g_vol * dz, axis=1)
    dm_gas     = float(mass_gas[-1] - mass_gas[0])

    mass_s_vol = np.sum(rho_s, axis=1)
    mass_solid = np.sum(mass_s_vol * dz, axis=1)
    dm_solid   = float(mass_solid[-1] - mass_solid[0])

    rho_g_in  = np.einsum('ti,i->t', C_in_safe, MW)
    rho_g_out = np.einsum('ti,i->t', C[:, :, -1], MW)
    flux_mass = float(np.trapz(vin * rho_g_in * epsi_r - vout * rho_g_out * epsi_r, t))

    imb_mass     = dm_gas + dm_solid - flux_mass
    denom_mass   = max(abs(dm_gas), abs(dm_solid), 1e-30)
    imb_mass_rel = imb_mass / denom_mass

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 2 — ESPECIES GAS
    # ══════════════════════════════════════════════════════════════════════════
    mol_i_gas    = epsi_r * np.sum(C * dz, axis=2)         # (n_t, nc)
    dmol_i       = mol_i_gas[-1] - mol_i_gas[0]            # (nc,)
    flux_mol_net = np.trapz(
        vin[:, None] * C_in_safe * epsi_r - vout[:, None] * C[:, :, -1] * epsi_r,
        t, axis=0,
    )                                                       # (nc,)
    fuente_rxn_i = dmol_i - flux_mol_net                   # (nc,)

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 3 — MASA SÓLIDA
    # ══════════════════════════════════════════════════════════════════════════
    comp_names  = ["biomasa", "char", "humedad"]
    mass_s_comp = np.sum(rho_s * dz, axis=2)               # (n_t, 3)
    dm_s_comp   = mass_s_comp[-1] - mass_s_comp[0]         # (3,)
    S_rxn_j     = dm_s_comp.copy()                         # conv_solid ≈ 0 for batch/cstr

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 4 — ENERGÍA GAS (Hg)
    # ODE: dHg/dt = −ε·div(Fh) − div(q_cond) − q_gs + q_wall + q_mt
    # ══════════════════════════════════════════════════════════════════════════
    dHg = float(np.sum(Hg * dz, axis=1)[-1] - np.sum(Hg * dz, axis=1)[0])

    T_in_safe = np.where(batch_mask, gas_T_ref, Tin)
    h_i_in    = calc_species_enthalpy(T_in_safe, prop_gas, nc, gas_T_ref)  # (nc, n_t)
    Hg_in_    = epsi_r * np.einsum('it,ti->t', h_i_in, C_in_safe)
    Hg_in_[batch_mask] = 0.0
    Fh_conv_net = float(np.trapz(vin * Hg_in_ - vout * Hg[:, -1], t))

    # Q_gs exacto desde acumulador ODE — ∫q_gs_vol dt integrado por BDF
    Q_gs = float(np.sum(gasifier._Q_gs_acc_results[-1] * dz))

    if shell_tube:
        h_wall_val  = (float(np.mean(trans_config["h_wall"]))
                       if trans_config.get("mode") == "constant" else 25.0)
        Q_wall      = float(np.trapz(
            np.sum(h_wall_val * (Pi / Ai) * (Tw - Tg) * dz, axis=1), t))
        q_wall_note = f"h_wall≈{h_wall_val:.0f} W/m²/K, shell_tube"
    else:
        h_wall_arr_v = np.full(nn, float(np.mean(trans_config["h_wall"]))
                               if trans_config.get("mode") == "constant" else 25.0)
        Qw_dot = np.zeros(len(t))
        for k in range(len(t)):
            _, qk, _ = wall_heat_flux(
                Tg=Tg[k], h_wall=h_wall_arr_v,
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )
            Qw_dot[k] = qk / Ai
        Q_wall      = float(np.trapz(Qw_dot, t))
        q_wall_note = thermal_bc_cfg["mode"]

    # Q_mt y Q_rxn exactos desde acumuladores ODE
    Q_mt_exact  = float(np.sum(gasifier._Q_mt_acc_results[-1]  * dz))
    Q_rxn_exact = float(np.sum(gasifier._Q_rxn_acc_results[-1] * dz))

    # Residual del gas: ΔHg = Fh_conv_net − Q_gs + Q_wall + Q_mt → ≈ 0 ★
    residual_gas     = dHg - Fh_conv_net + Q_gs - Q_wall - Q_mt_exact
    _ref_energy      = max(abs(Q_wall), abs(Q_gs), abs(Fh_conv_net), abs(dHg), 1e-30)
    residual_gas_rel = residual_gas / _ref_energy

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 5 — ENERGÍA SÓLIDO
    # ΔHs usando H_j(Ts) = ∫_{273}^{Ts} Cp_j dT (integral exacta, no Cp·Ts proxy)
    # Con esta definición: d(Σ ρ_j·H_j)/dt = Q_rxn + q_gs exactamente → ★ cierre
    # ══════════════════════════════════════════════════════════════════════════
    def _Hs_true(rs, Ts_):
        """Σ_j ρ_j · H_j(Ts)  donde H_j = ∫_{T_ref}^{Ts} Cp_j dT [J/m³_bed]."""
        Ts_a = np.asarray(Ts_, float)
        res  = np.zeros(len(Ts_a), dtype=float)
        for j in range(3):
            res += rs[j] * np.asarray(h_fns[j](Ts_a), float)
        return res  # (N,) [J/m³_bed]

    dHs = (float(np.sum(_Hs_true(rho_s[-1], Ts[-1]) * dz))
           - float(np.sum(_Hs_true(rho_s[0],  Ts[0])  * dz)))

    Q_gs_solid     = Q_gs                                  # sólido gana lo que gas pierde
    residual_solid = dHs - Q_gs_solid - Q_rxn_exact        # ≈ 0 ★ (con H_j integral + Q_rxn acum.)
    _ref_solid     = max(abs(dHs), abs(Q_gs_solid), abs(Q_rxn_exact), 1e-30)
    residual_solid_rel = residual_solid / _ref_solid

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO 6 — ENERGÍA PARED (Tw) — solo si shell_tube
    # ══════════════════════════════════════════════════════════════════════════
    energy_wall    = None
    dHw = Q_gw = residual_wall = residual_wall_rel = 0.0
    if shell_tube:
        wall_config = params["wall_config"]
        mat = wall_config["material"]
        A_w = float(wall_config["A_w"])
        rho_w_f = eval_solid_property(mat["rho"], Tw[-1])
        Cp_w_f  = eval_solid_property(mat["cp"],  Tw[-1])
        rho_w_i = eval_solid_property(mat["rho"], Tw[0])
        Cp_w_i  = eval_solid_property(mat["cp"],  Tw[0])
        dHw = float(np.sum(
            (rho_w_f * Cp_w_f * Tw[-1] - rho_w_i * Cp_w_i * Tw[0]) * A_w * dz) / Ai)
        h_w_v = (float(np.mean(trans_config["h_wall"]))
                 if trans_config.get("mode") == "constant" else 25.0)
        Q_gw = float(np.trapz(
            h_w_v * Pi * np.sum((Tg - Tw) * dz, axis=1) / Ai, t))
        residual_wall     = dHw - Q_gw
        residual_wall_rel = residual_wall / max(abs(dHw), 1e-30)
        energy_wall = {"dHw": dHw * _norm, "Q_gw": Q_gw * _norm,
                       "residual": residual_wall * _norm, "residual_rel": residual_wall_rel}

    # ══════════════════════════════════════════════════════════════════════════
    # NORMALIZACIÓN: todos los resultados [J/m²] → [J/m³_bed] dividiendo por L
    # ══════════════════════════════════════════════════════════════════════════
    # Masa [kg/m²] → [kg/m³_bed]
    dm_gas_n    = dm_gas    * _norm
    dm_solid_n  = dm_solid  * _norm
    flux_mass_n = flux_mass * _norm
    imb_mass_n  = imb_mass  * _norm
    dm_s_comp_n = dm_s_comp * _norm
    S_rxn_j_n   = S_rxn_j  * _norm

    # Moles [mol/m²] → [mol/m³_bed]
    dmol_i_n       = dmol_i       * _norm
    flux_mol_net_n = flux_mol_net * _norm
    fuente_rxn_i_n = fuente_rxn_i * _norm

    # Energía [J/m²] → [J/m³_bed]
    dHg_n          = dHg          * _norm
    Fh_conv_net_n  = Fh_conv_net  * _norm
    Q_gs_n         = Q_gs         * _norm
    Q_wall_n       = Q_wall       * _norm
    Q_mt_exact_n   = Q_mt_exact   * _norm
    Q_rxn_exact_n  = Q_rxn_exact  * _norm
    residual_gas_n = residual_gas * _norm
    dHs_n          = dHs          * _norm
    Q_gs_solid_n   = Q_gs_solid   * _norm
    residual_sol_n = residual_solid * _norm
    dHw_n          = dHw          * _norm
    Q_gw_n         = Q_gw         * _norm
    residual_wall_n = residual_wall * _norm

    # ══════════════════════════════════════════════════════════════════════════
    # FORMATO DE SALIDA
    # ══════════════════════════════════════════════════════════════════════════
    lines = [
        "",
        "  " + "═" * W,
        f"  GASIFICADOR — BALANCE DE CONSERVACIÓN",
        f"  modo={bc_mode}   N={nn}   L={L:.3f} m   ε={epsi_r}   "
        f"t={t[0]:.1f}→{t[-1]:.1f} s",
        f"  Integrado sobre volumen del lecho y tiempo. Unidades: [kg/m³], [mol/m³], [J/m³].",
        "  " + "═" * W,
    ]

    # ─── TABLA: MASA ───────────────────────────────────────────────────────────
    lines += [
        _sep("MASA [kg/m³]  ·  ★ = cierre numérico (Residual ≈ 0)"),
        _hdr(),
        "  " + "─" * W,
        _trow("  m_gas",    dm_gas_n,   0.0,          dm_gas_n,  note="(→ reacciones)"),
        _trow("  m_sólido", dm_solid_n, 0.0,          dm_solid_n, note="(reacciones →)"),
        _trow("★ m_total",  dm_gas_n + dm_solid_n, flux_mass_n, imb_mass_n,
              imb_mass_rel * 100),
        "  " + "─" * W,
    ]

    # ─── TABLA: ESPECIES GAS ──────────────────────────────────────────────────
    lines += [
        _sep("~ ESPECIES GAS [mol/m³]  ·  Residual físico ≠ 0 esperado (moles producidos/consumidos por rxns)"),
        _hdr(),
        "  " + "─" * W,
    ]
    for i, sp in enumerate(species):
        note = ("producido" if fuente_rxn_i[i] > 1e-14
                else "consumido" if fuente_rxn_i[i] < -1e-14 else "")
        lines.append(_trow(f"  N_{sp}", dmol_i_n[i], flux_mol_net_n[i],
                           fuente_rxn_i_n[i], note=note))
    lines += [
        "  " + "─" * W,
        _trow("  N_total_gas", float(np.sum(dmol_i_n)),
              float(np.sum(flux_mol_net_n)), float(np.sum(fuente_rxn_i_n))),
        "  " + "─" * W,
    ]

    # ─── TABLA: SÓLIDO ────────────────────────────────────────────────────────
    vs_val = float(params["bc_config"].get("v_solid", 0.0))
    lines += [
        _sep("~ SÓLIDO [kg/m³]  ·  Residual físico ≠ 0 esperado (masa transformada por reacciones)"
             + (f"  (vs≠0: conv. sólida incluida en Residual)" if abs(vs_val) > 1e-12 else "")),
        _hdr(),
        "  " + "─" * W,
    ]
    for j, name in enumerate(comp_names):
        note = ("consumido" if S_rxn_j[j] < -1e-10
                else "generado" if S_rxn_j[j] > 1e-10 else "")
        lines.append(_trow(f"  m_{name}", float(dm_s_comp_n[j]),
                           0.0, float(S_rxn_j_n[j]), note=note))
    lines += ["  " + "─" * W]

    if shell_tube:
        lines += [
            _sep("PARED [J/m³]  ·  ★ cierre numérico  (Residual ≈ Q_ext + Q_ax)"),
            _hdr(),
            "  " + "─" * W,
            _trow("★ Hw", dHw_n, Q_gw_n, residual_wall_n, residual_wall_rel * 100),
            "  " + "─" * W,
        ]

    # ─── VEREDICTO ────────────────────────────────────────────────────────────
    verdict = _ok(imb_mass_rel)
    lines += [
        "",
        "  " + "═" * W,
        f"  ★ Cierre numérico  m_total:  {imb_mass_rel*100:>+.3f} %   "
        f"{verdict}   (umbral ±1 %)",
    ]
    if shell_tube:
        lines.append(
            f"  ★ Cierre numérico  Hw:       {residual_wall_rel*100:>+.3f} %   "
            f"{_ok(residual_wall_rel)}"
        )

    # ─── DESGLOSE ENERGÉTICO ──────────────────────────────────────────────────
    cond_note = "= 0 exacto" if nn == 1 else "incluido en cierre (N>1)"

    lines += [
        "",
        _sep("DESGLOSE ENERGÉTICO [J/m³]"),
        "",
        "  Gas (Hg):  dHg/dt = −ε·div(Fh) − div(q_cond) − q_gs + q_wall + q_mt",
        f"    ΔHg                    = {dHg_n:>+14.4e}  [acumulación gas]",
        f"    Fh_conv_neto           = {Fh_conv_net_n:>+14.4e}  [ε·(Fh_in−Fh_out), exacto]",
        f"    q_cond_gas_neto        = {'0':>14}  [{cond_note}]",
        f"    Q_gs (gas→sólido)      = {Q_gs_n:>+14.4e}  [acum. ODE — exacto]",
        f"    Q_wall (pared→gas)     = {Q_wall_n:>+14.4e}  [{q_wall_note}]",
        f"    Q_mt  (sól→gas, exacto)= {Q_mt_exact_n:>+14.4e}  [acum. ODE, exacto]",
        f"    ─{'─'*50}",
        f"  ★ Cierre_Hg              = {residual_gas_n:>+14.4e}  "
        f"rel={residual_gas_rel*100:+.2f} %   {_ok(residual_gas_rel)}",
        "",
        "  Sólido (Ts):  d(Σ ρ_j·H_j)/dt = Q_rxn + q_gs  (H_j = ∫Cp_j dT, exacto)",
        f"    ΔHs [∫Cp dT, exacto]  = {dHs_n:>+14.4e}  [Σ ρ_j·H_j, integral de Cp]",
        f"    Q_gs_al_sólido         = {Q_gs_solid_n:>+14.4e}  [acum. ODE — exacto]",
        f"    Q_rxn (exacto)         = {Q_rxn_exact_n:>+14.4e}  [acum. ODE: ∫Q_rxn_vol dt]",
        f"    ─{'─'*50}",
        f"  ★ Cierre_Hs (ΔHs−Q_gs−Q_rxn) = {residual_sol_n:>+14.4e}  "
        f"rel={residual_solid_rel*100:+.2f} %   {_ok(residual_solid_rel)}",
    ]

    # ─── BALANCE GLOBAL DE ENERGÍA ────────────────────────────────────────────
    global_lhs      = dHg + dHs
    global_rhs      = Fh_conv_net + Q_wall + Q_rxn_exact + Q_mt_exact
    global_residual = global_lhs - global_rhs
    global_ref      = max(abs(global_lhs), abs(Q_wall), 1e-30)
    global_rel      = global_residual / global_ref
    global_ok       = "✓ OK" if abs(global_rel) < 0.01 else "✗ REVISAR"

    lines += [
        "",
        _sep("BALANCE GLOBAL ENERGÍA [J/m³]  ★ = cierre numérico"),
        "    Ecuación: ΔHg + ΔHs = Fh_neto + Q_wall + Q_rxn + Q_mt",
        f"    ΔHg + ΔHs             = {(dHg_n + dHs_n):>+14.4e}  [acumulación total]",
        f"    Fh_conv_neto          = {Fh_conv_net_n:>+14.4e}  [exacto]",
        f"    Q_wall                = {Q_wall_n:>+14.4e}  [{q_wall_note}]",
        f"    Q_rxn (exacto)        = {Q_rxn_exact_n:>+14.4e}  [acum. ODE]",
        f"    Q_mt  (exacto)        = {Q_mt_exact_n:>+14.4e}  [acum. ODE]",
        f"    ─{'─'*50}",
        f"  ★ Cierre global         = {global_residual * _norm:>+14.4e}  "
        f"rel={global_rel*100:+.2f} %   {global_ok}",
        "",
        "  " + "═" * W,
        "",
    ]

    report = "\n".join(lines)
    if verbose:
        print(report)

    return {
        "mass_total": {
            "dm_gas": dm_gas_n, "dm_solid": dm_solid_n,
            "flux_mass": flux_mass_n,
            "imbalance": imb_mass_n, "imbalance_rel": imb_mass_rel,
        },
        "species_gas": {
            "species": species,
            "dmol_i": dmol_i_n, "flux_mol_net": flux_mol_net_n,
            "fuente_rxn_i": fuente_rxn_i_n,
        },
        "solid": {
            "comp_names": comp_names,
            "dm_s_comp": dm_s_comp_n, "S_rxn_j": S_rxn_j_n,
        },
        "energy_gas": {
            "dHg": dHg_n, "Fh_conv_net": Fh_conv_net_n,
            "Q_gs": Q_gs_n, "Q_wall": Q_wall_n,
            "Q_mt_exact": Q_mt_exact_n,
            "closure": residual_gas_n, "closure_rel": residual_gas_rel,
        },
        "energy_solid": {
            "dHs": dHs_n, "Q_gs_solid": Q_gs_solid_n,
            "Q_rxn_exact": Q_rxn_exact_n,
            "closure": residual_sol_n, "closure_rel": residual_solid_rel,
        },
        "energy_global": {
            "dHg_plus_dHs": dHg_n + dHs_n,
            "Fh_conv_net":  Fh_conv_net_n,
            "Q_wall":       Q_wall_n,
            "Q_rxn_exact":  Q_rxn_exact_n,
            "Q_mt_exact":   Q_mt_exact_n,
            "closure":      global_residual * _norm,
            "closure_rel":  global_rel,
        },
        "energy_wall": energy_wall,
        "report": report,
    }


# ─── Convenience print ───────────────────────────────────────────────────────

def print_summary(gasifier, params: dict) -> None:
    """Informe de balances completo. Delegado a check_balances."""
    check_balances(gasifier, params, verbose=True)


# ─── Tabular display ─────────────────────────────────────────────────────────

def display_balances(balances: dict) -> None:
    """
    Display the dict returned by check_balances() as formatted DataFrames.

    Prints one DataFrame per balance section. Adds Estado column to closure
    rows: ✓ OK when |rel| < 1 %, ⚠ REVISAR otherwise.

    Parameters
    ----------
    balances : dict
        Return value of check_balances(gasifier, params, verbose=False).
    """
    import numpy as np
    import pandas as pd
    try:
        from IPython.display import display as _display
    except ImportError:
        _display = print

    def _estado(rel: float) -> str:
        return "✓ OK" if abs(rel) < 0.01 else "⚠ REVISAR"

    def _fmt(x):
        return f"{float(x):+.4e}" if isinstance(x, (int, float, np.floating)) else str(x)

    # ── ★ Masa total ──────────────────────────────────────────────────────────
    mb = balances["mass_total"]
    rel_m = float(mb["imbalance_rel"])
    df_m = pd.DataFrame([
        {"Término":               "Δm_gas",          "Valor [kg/m³]": _fmt(mb["dm_gas"]),    "Tipo": ""},
        {"Término":               "Δm_sólido",        "Valor [kg/m³]": _fmt(mb["dm_solid"]),  "Tipo": ""},
        {"Término":               "Flujo_masa_neto",  "Valor [kg/m³]": _fmt(mb["flux_mass"]), "Tipo": ""},
        {"Término": "★ Cierre",  "Valor [kg/m³]": _fmt(mb["imbalance"]),
         "Tipo": f"{rel_m*100:.3f} %  {_estado(rel_m)}"},
    ]).set_index("Término")
    print("── ★ Masa total [kg/m³] ──────────────────────────────────────────────")
    _display(df_m)

    # ── ✗ Especies gas ────────────────────────────────────────────────────────
    sp = balances["species_gas"]
    df_sp = pd.DataFrame({
        "ΔC_i [mol/m³]":         np.asarray(sp["dmol_i"],       float),
        "Flujo_neto [mol/m³]":   np.asarray(sp["flux_mol_net"], float),
        "~ Fuente_rxn [mol/m³]": np.asarray(sp["fuente_rxn_i"], float),
    }, index=sp["species"])
    df_sp = df_sp.applymap(_fmt)
    print("\n── ~ Especies gas [mol/m³]  (Fuente_rxn = producida/consumida por reacciones — residual físico esperado ≠ 0) ─")
    _display(df_sp)

    # ── ✗ Masa sólida por componente ──────────────────────────────────────────
    if balances.get("solid"):
        sd = balances["solid"]
        df_sd = pd.DataFrame({
            "Δρ_j [kg/m³]":     np.asarray(sd["dm_s_comp"], float),
            "~ S_rxn [kg/m³]":  np.asarray(sd["S_rxn_j"],   float),
        }, index=sd["comp_names"])
        df_sd = df_sd.applymap(_fmt)
        print("\n── ~ Masa sólida [kg/m³]  (S_rxn = masa transformada por reacciones — residual físico esperado ≠ 0) ──────")
        _display(df_sd)

    # ── ★ Energía gas ─────────────────────────────────────────────────────────
    eg = balances["energy_gas"]
    rel_eg = float(eg["closure_rel"])
    df_eg = pd.DataFrame([
        {"Término": "ΔHg",                   "Valor [J/m³]": _fmt(eg["dHg"]),         "Nota": "acumulación gas"},
        {"Término": "Fh_conv_neto",          "Valor [J/m³]": _fmt(eg["Fh_conv_net"]), "Nota": "exacto"},
        {"Término": "Q_gs  (gas→sólido)",    "Valor [J/m³]": _fmt(eg["Q_gs"]),        "Nota": "acum. ODE — exacto"},
        {"Término": "Q_wall (pared→gas)",    "Valor [J/m³]": _fmt(eg["Q_wall"]),      "Nota": "thermal_bc"},
        {"Término": "Q_mt  (sól→gas)",       "Valor [J/m³]": _fmt(eg["Q_mt_exact"]), "Nota": "acum. ODE — exacto"},
        {"Término": "★ Cierre_Hg",           "Valor [J/m³]": _fmt(eg["closure"]),
         "Nota": f"{rel_eg*100:.3f} %  {_estado(rel_eg)}"},
    ]).set_index("Término")
    print("\n── ★ Energía gas  Hg [J/m³] ─────────────────────────────────────────────")
    _display(df_eg)

    # ── ★ Energía sólido ──────────────────────────────────────────────────────
    es = balances["energy_solid"]
    rel_es = float(es["closure_rel"])
    df_es = pd.DataFrame([
        {"Término": "ΔHs  [∫Cp dT, exacto]", "Valor [J/m³]": _fmt(es["dHs"]),           "Nota": "Σ ρⱼ·H_j(Ts)"},
        {"Término": "Q_gs  (gas→sólido)",     "Valor [J/m³]": _fmt(es["Q_gs_solid"]),    "Nota": "acum. ODE — exacto"},
        {"Término": "Q_rxn (exacto)",         "Valor [J/m³]": _fmt(es["Q_rxn_exact"]),   "Nota": "acum. ODE — exacto"},
        {"Término": "★ Cierre_Hs",            "Valor [J/m³]": _fmt(es["closure"]),
         "Nota": f"{rel_es*100:.3f} %  {_estado(rel_es)}"},
    ]).set_index("Término")
    print("\n── ★ Energía sólido  Ts [J/m³] ──────────────────────────────────────────")
    _display(df_es)

    # ── ★ Balance global de energía ──────────────────────────────────────────
    eg_glob = balances.get("energy_global")
    if eg_glob is not None:
        rel_glob = float(eg_glob["closure_rel"])
        df_glob = pd.DataFrame([
            {"Término": "ΔHg + ΔHs",       "Valor [J/m³]": _fmt(eg_glob["dHg_plus_dHs"]), "Nota": "acumulación total"},
            {"Término": "Fh_conv_neto",     "Valor [J/m³]": _fmt(eg_glob["Fh_conv_net"]),  "Nota": "exacto"},
            {"Término": "Q_wall",           "Valor [J/m³]": _fmt(eg_glob["Q_wall"]),        "Nota": "thermal_bc"},
            {"Término": "Q_rxn (exacto)",   "Valor [J/m³]": _fmt(eg_glob["Q_rxn_exact"]),   "Nota": "acum. ODE"},
            {"Término": "Q_mt  (exacto)",   "Valor [J/m³]": _fmt(eg_glob["Q_mt_exact"]),    "Nota": "acum. ODE"},
            {"Término": "★ Cierre global",  "Valor [J/m³]": _fmt(eg_glob["closure"]),
             "Nota": f"{rel_glob*100:.3f} %  {_estado(rel_glob)}"},
        ]).set_index("Término")
        print("\n── ★ Balance global  Hg+Hs [J/m³] ──────────────────────────────────────")
        _display(df_glob)

    # ── ★ Energía pared (solo shell-tube) ─────────────────────────────────────
    ew = balances.get("energy_wall")
    if ew:
        rel_ew = float(ew.get("closure_rel", 0.0))
        rows_ew = [
            {"Término": k, "Valor [J/m³]": _fmt(v), "Nota": ""}
            for k, v in ew.items() if k not in ("closure", "closure_rel")
        ]
        rows_ew.append({
            "Término": "★ Cierre_Hw",
            "Valor [J/m³]": _fmt(ew.get("closure", 0.0)),
            "Nota": f"{rel_ew*100:.3f} %  {_estado(rel_ew)}",
        })
        df_ew = pd.DataFrame(rows_ew).set_index("Término")
        print("\n── ★ Energía pared  Tw [J/m³] ────────────────────────────────────────────")
        _display(df_ew)
