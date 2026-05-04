"""
Balance de conservación para el reactor tubular 1D.

Todos los balances usan integración trapezoidal sobre tiempo y espacio.
Unidades de presentación: [kg/m³_bed], [mol/m³_bed], [J/m³_bed]
(normalizadas por L = dz·N para comparabilidad entre reactores de distinto tamaño).

Convención de signos:
    Acumulación  >0  →  la variable aumentó neta en el volumen de control
    Flujo neto   >0  →  entra más de lo que sale (convención consistente con gasificador)
    q_gs         >0  →  el gas cede calor al sólido (gas pierde, sólido gana)

Clasificación de residuales:
    ★ Cierre numérico  →  debe ser ≈ 0 (umbral 1%)
    Fuente física      →  espera ser ≠ 0 (producción/consumo por reacciones)

Diferencias respecto al gasificador:
    - Sin rho_solid en sv (catalizador constante → sin transporte de masa sólido→gas)
    - Sin q_masstransfer (no se crean moléculas desde el sólido)
    - Sin thermal_mass_correction (rho_cat = cte)
    - Q_rxn recuperable post-hoc llamando rate_fn sobre resultados almacenados
    - Sin acumuladores ODE (todos los términos son recuperables o aproximables)
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.physics.thermal.wall_heat_flux  import wall_heat_flux


R_GAS = 8.31446261815324   # [J/mol/K]
_TOL  = 1.0e-2             # umbral 1% para cierre numérico


def check_balances(reactor, params: dict, verbose: bool = True) -> dict:
    """
    Informe completo de conservación término a término para el reactor.

    Secciones:
        1. Masa total del gas               (★ sin reacciones; fuente física con rxn)
        2. Especies gaseosas por especie    (★ sin reacciones; fuente física con rxn)
        3. Energía del gas — Hg             (★ aproximado via q_gs y q_wall)
        4. Energía del sólido — Ts          (★ si has_catalyst y sin reacciones)
        5. Energía de la pared — Tw         (★ si shell_tube)

    Todos los valores normalizados por L [J/m³_bed] o [mol/m³_bed] o [kg/m³_bed].

    Parameters
    ----------
    reactor : SimpleNamespace  — output de build_reactor_results()
    params  : dict
    verbose : bool             — True imprime el informe

    Returns
    -------
    dict con claves: mass_gas, species_gas, energy_gas, energy_solid, energy_wall
    """
    # ── Parámetros ─────────────────────────────────────────────────────────────
    nc           = int(params["n_comp"])
    nn           = int(params["N"])
    dz           = float(params["dz"])
    epsi         = float(params["epsi"])
    Ai           = float(params["Ai"])
    Pi           = float(params["Pi"])
    Po           = float(params["Po"])
    gas_T_ref    = float(params["gas_T_ref"])
    MW           = np.asarray(params["MW"], dtype=float)   # (nc,) [kg/mol]
    prop_gas     = params["prop_gas"]
    thermal_bc   = params["thermal_bc_config"]
    trans_config = params["trans_config"]
    species      = list(params["species"])
    has_catalyst = params.get("catalyst_config") is not None
    shell_tube   = params.get("wall_config") is not None
    reactions    = params.get("reactions_config", [])
    has_rxn      = len(reactions) > 0

    L = dz * nn   # longitud total [m]

    # ── Resultados ─────────────────────────────────────────────────────────────
    t    = reactor._t_results                           # (n_t,)
    C    = reactor._C_results                           # (n_t, nc, N)
    Hg   = reactor._Hg_results                         # (n_t, N)
    Tg   = reactor._Tg_results                         # (n_t, N)
    Ts   = reactor._Ts_results                         # (n_t, N) or None
    Tw   = reactor._Tw_results                         # (n_t, N) or None
    P    = reactor._P_results                          # (n_t, N) [bar]
    vin  = reactor._v_in_results                       # (n_t,)
    vout = reactor._v_out_results                      # (n_t,)
    Cin  = reactor._C_in_results                       # (n_t, nc) NaN=batch
    Tin  = reactor._T_in_results                       # (n_t,)   NaN=batch

    is_batch = np.all(vin == 0.0)

    Cin_safe = np.nan_to_num(Cin, nan=0.0)

    # Factor de normalización: [valor/m²_section] → [valor/m³_bed]
    _inv_L = 1.0 / L

    # ── Helpers de formato ─────────────────────────────────────────────────────
    W = 80

    def _line(ch="-"):
        return ch * W

    def _row(label, acum, flux, resid, pct=None, star=False, note=""):
        prefix = "★" if star else " "
        pct_s  = f"{pct:>+7.2f}%" if pct is not None else "       —"
        return (f"  {prefix} {label:<22} {acum:>+14.4e}  {flux:>+14.4e}"
                f"  {resid:>+14.4e}  {pct_s}  {note}")

    def _hdr():
        return (f"  {'Variable':<23}  {'Acumulacion':>14}  {'Flujo neto':>14}"
                f"  {'Residual':>14}  {'Res [%]':>8}  Nota")

    def _pct(resid, acum, flux):
        denom = max(abs(acum), abs(flux), 1.0e-30)
        return 100.0 * abs(resid) / denom

    lines = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. MASA TOTAL DEL GAS  [kg/m³_bed]
    # ══════════════════════════════════════════════════════════════════════════
    # m_gas = epsi * Σ_i C_i * MW_i  [kg/m³_bed] por celda
    m_gas_field = epsi * np.tensordot(C, MW, axes=([1], [0]))  # (n_t, N) [kg/m³_bed]
    m_gas_mean  = np.sum(m_gas_field * dz, axis=1) * _inv_L    # (n_t,)   [kg/m³_bed]
    delta_m_gas = float(m_gas_mean[-1] - m_gas_mean[0])

    # Flujo convectivo de masa [kg/m³_bed]: ∫ (v_in * Σ C_in * MW - v_out * Σ C_out * MW) * epsi dt / L
    m_flux_in  = vin * np.dot(Cin_safe, MW) * epsi * _inv_L    # (n_t,) [kg/m³_bed/s]
    m_flux_out = vout * np.dot(C[:, :, -1], MW) * epsi * _inv_L
    flux_m_gas = float(np.trapz(m_flux_in - m_flux_out, t))

    resid_m_gas = delta_m_gas - flux_m_gas
    pct_m_gas   = _pct(resid_m_gas, delta_m_gas, flux_m_gas)
    note_m_gas  = "fuente rxn" if has_rxn else ("★" if pct_m_gas < 1.0 else "✗ >1%")

    mass_gas = {
        "delta_kg_m3":   delta_m_gas,
        "flux_kg_m3":    flux_m_gas,
        "residual":      resid_m_gas,
        "residual_pct":  pct_m_gas,
    }

    lines += [
        "", _line("═"),
        f"  MASA GAS  [kg/m³_bed]" + ("  (con reacciones — residual = masa reaccionada)" if has_rxn else "  ★"),
        _line(),
        _hdr(), _line(),
        _row("m_gas total", delta_m_gas, flux_m_gas, resid_m_gas, pct_m_gas,
             star=not has_rxn, note=note_m_gas),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 2. ESPECIES GASEOSAS  [mol/m³_bed]
    # ══════════════════════════════════════════════════════════════════════════
    lines += ["", _line("═"),
              f"  ESPECIES GAS  [mol/m³_bed]" + ("  (residual = moles producidos/consumidos por rxn)" if has_rxn else "  ★"),
              _line(), _hdr(), _line()]

    species_gas = {}
    for i, sp in enumerate(species):
        mol_field  = epsi * C[:, i, :]                          # (n_t, N)
        mol_mean   = np.sum(mol_field * dz, axis=1) * _inv_L   # (n_t,)
        delta_mol  = float(mol_mean[-1] - mol_mean[0])

        flux_in_i  = vin  * Cin_safe[:, i] * epsi * _inv_L
        flux_out_i = vout * C[:, i, -1]    * epsi * _inv_L
        flux_mol   = float(np.trapz(flux_in_i - flux_out_i, t))

        resid = delta_mol - flux_mol
        pct   = _pct(resid, delta_mol, flux_mol)
        star  = not has_rxn
        note  = ("★" if pct < 1.0 else "✗") if star else "fuente rxn"
        species_gas[sp] = {"delta": delta_mol, "flux": flux_mol,
                           "residual": resid, "residual_pct": pct}
        lines.append(_row(f"C_{sp}", delta_mol, flux_mol, resid, pct, star=star, note=note))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ENERGÍA GAS — Hg  [J/m³_bed]
    # ══════════════════════════════════════════════════════════════════════════
    Hg_mean  = np.sum(Hg * dz, axis=1) * _inv_L    # (n_t,) [J/m³_bed]
    delta_Hg = float(Hg_mean[-1] - Hg_mean[0])

    # Flujo entálpico convectivo en inlet y outlet [J/m³_bed/s]
    # Fh = v * Σ C_i * h_i(Tg)   — sin factor epsi (la norma del framework)
    h_in_k  = np.zeros(len(t))   # [J/m²/s]
    h_out_k = np.zeros(len(t))
    for k in range(len(t)):
        if not is_batch:
            T_in_k = Tin[k] if not np.isnan(Tin[k]) else float(Tg[k, 0])
            hi_in  = calc_species_enthalpy(
                np.full(nn, T_in_k), prop_gas, nc, gas_T_ref)   # (nc, N)
            h_in_k[k]  = vin[k]  * np.dot(Cin_safe[k], hi_in[:, 0])
            hi_out = calc_species_enthalpy(
                Tg[k, -1:], prop_gas, nc, gas_T_ref)             # (nc, 1)
            h_out_k[k] = vout[k] * np.dot(C[k, :, -1], hi_out[:, 0])

    Fh_neto = float(np.trapz((h_in_k - h_out_k) * epsi * _inv_L, t))

    # Flujo pared  q_wall [J/m³_bed]: desde thermal_bc_config
    if shell_tube:
        # Con pared dinámica: ∫ h_wall * (Pi/Ai) * (Tw - Tg) dz dt / L
        h_wall_val = (trans_config["h_wall"][0]
                      if trans_config["h_wall"] is not None else 10.0)
        q_wall_field = h_wall_val * (Pi / Ai) * (Tw - Tg)   # (n_t, N) [W/m³_bed]
        Q_wall_total = float(np.trapz(
            np.sum(q_wall_field * dz, axis=1) * _inv_L, t))
    else:
        q_wall_total_list = []
        for k in range(len(t)):
            h_w_k  = (trans_config["h_wall"][0]
                      if trans_config["h_wall"] is not None else 10.0)
            qw_k, _, _ = wall_heat_flux(
                Tg=Tg[k], h_wall=np.full(nn, h_w_k),
                thermal_bc_config=thermal_bc,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )
            q_wall_total_list.append(np.sum(qw_k * dz) * _inv_L)
        Q_wall_total = float(np.trapz(q_wall_total_list, t))

    # Flujo gas↔sólido q_gs [J/m³_bed]: h_bed * a_p * (Tg - Ts)
    if has_catalyst:
        cat = params["catalyst_config"]
        a_p     = float(cat["a_p"])
        h_bed_v = (trans_config["h_bed"][0]
                   if trans_config["h_bed"] is not None else 50.0)
        q_gs_field = h_bed_v * a_p * (Tg - Ts)   # (n_t, N) [W/m³_bed]
        Q_gs_total = float(np.trapz(
            np.sum(q_gs_field * dz, axis=1) * _inv_L, t))
    else:
        Q_gs_total = 0.0

    # Q_rxn_gas (reacciones homogéneas → calor directo al gas) [J/m³_bed]
    Q_rxn_gas_total = 0.0
    for rxn in reactions:
        if rxn.get("type", "").lower() == "homogeneous":
            P_Pa = P * 1.0e5
            for k in range(len(t)):
                rate_k = rxn["rate_fn"](C[k], Tg[k], Ts[k] if Ts is not None else None,
                                        P_Pa[k], params)
                dH  = rxn["dH_rxn"]
                dHv = float(dH(Tg[k]).mean()) if callable(dH) else float(dH)
                Q_rxn_gas_total += float(np.sum((-dHv) * rate_k * dz)) * _inv_L * (t[1] - t[0])

    # dHg/dt = -epsi * div(Fh) - q_gs + q_wall + q_rxn_gas
    # → ΔHg = Fh_neto - Q_gs + Q_wall + Q_rxn_gas    (sign: Fh_neto = in-out > 0 si gana)
    Hg_pred = Fh_neto - Q_gs_total + Q_wall_total + Q_rxn_gas_total
    resid_Hg = delta_Hg - Hg_pred
    pct_Hg   = _pct(resid_Hg, delta_Hg, Hg_pred)

    energy_gas = {
        "delta_Hg":      delta_Hg,
        "Fh_neto":       Fh_neto,
        "Q_gs":          Q_gs_total,
        "Q_wall":        Q_wall_total,
        "Q_rxn_gas":     Q_rxn_gas_total,
        "residual":      resid_Hg,
        "residual_pct":  pct_Hg,
    }

    lines += [
        "", _line("═"),
        "  ENERGÍA GAS — Hg  [J/m³_bed]  (cierre aproximado: q_gs desde h_bed constante)",
        _line(),
        f"  {'ΔHg (acumulacion)':<30} = {delta_Hg:>+16.4e}  [J/m³_bed]",
        f"  {'Fh_conv_neto (in−out)':<30} = {Fh_neto:>+16.4e}  [J/m³_bed]",
        f"  {'Q_gs (gas→solido)':<30} = {-Q_gs_total:>+16.4e}  [J/m³_bed]  (signo: gas pierde)",
        f"  {'Q_wall (pared→gas)':<30} = {Q_wall_total:>+16.4e}  [J/m³_bed]",
        f"  {'Q_rxn_gas (homo rxn→gas)':<30} = {Q_rxn_gas_total:>+16.4e}  [J/m³_bed]",
        _line(),
        f"  {'Cierre Hg (ΔHg − pred)':<30} = {resid_Hg:>+16.4e}  "
        f"rel={pct_Hg:+.2f}%  {'★ OK' if pct_Hg < 1.0 else 'aprox'}",
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ENERGÍA SÓLIDO — Ts  [J/m³_bed]  (solo si has_catalyst)
    # ══════════════════════════════════════════════════════════════════════════
    energy_solid = None
    if has_catalyst:
        cat        = params["catalyst_config"]
        rho_cat    = float(cat["rho_bulk"])
        Cp_fn      = cat["Cp_fn"]
        ind_config = params.get("induction_config")

        # ΔHs ≈ ∫ rho_cat * Cp_cat(Ts) * (Ts_f − Ts_0) dz / L   (catalizador constante)
        Cp_0  = np.asarray(Cp_fn(Ts[0]),  dtype=float)   # (N,)
        Cp_f  = np.asarray(Cp_fn(Ts[-1]), dtype=float)
        Cp_m  = 0.5 * (Cp_0 + Cp_f)                      # media trapezoidal (N,)
        delta_Hs = float(np.sum(rho_cat * Cp_m * (Ts[-1] - Ts[0]) * dz) * _inv_L)

        # Q_gs → sólido: opuesto al gas (sólido gana lo que gas pierde)
        Q_gs_to_solid = Q_gs_total   # [J/m³_bed]

        # Q_rxn_solid (reacciones heterogéneas → calor al sólido)
        Q_rxn_solid_total = 0.0
        P_Pa = P * 1.0e5
        for rxn in reactions:
            if rxn.get("type", "").lower() == "heterogeneous":
                for k in range(len(t)):
                    rate_k = rxn["rate_fn"](C[k], Tg[k], Ts[k], P_Pa[k], params)
                    dH  = rxn["dH_rxn"]
                    dHv = float(dH(Ts[k]).mean()) if callable(dH) else float(dH)
                    Q_rxn_solid_total += float(np.sum((-dHv) * rate_k * dz)) * _inv_L * (t[1] - t[0])

        # q_induction
        Q_ind_total = 0.0
        if ind_config is not None:
            for k in range(len(t)):
                if ind_config.get("mode") == "profile":
                    z_cells = (np.arange(nn) + 0.5) * dz
                    q_ind_k = np.asarray(ind_config["q_fn"](z_cells, float(t[k])))
                else:
                    q_ind_k = np.full(nn, float(ind_config.get("q_vol", 0.0)))
                Q_ind_total += float(np.sum(q_ind_k * dz)) * _inv_L * (t[1] - t[0])

        Hs_pred  = Q_gs_to_solid + Q_rxn_solid_total + Q_ind_total
        resid_Hs = delta_Hs - Hs_pred
        pct_Hs   = _pct(resid_Hs, delta_Hs, Hs_pred)

        energy_solid = {
            "delta_Hs":         delta_Hs,
            "Q_gs":             Q_gs_to_solid,
            "Q_rxn_solid":      Q_rxn_solid_total,
            "Q_induction":      Q_ind_total,
            "residual":         resid_Hs,
            "residual_pct":     pct_Hs,
        }
        lines += [
            "", _line("═"),
            "  ENERGÍA SÓLIDO — Ts  [J/m³_bed]  (Cp medio trapezoidal; sin thermal_correction)",
            _line(),
            f"  {'ΔHs (rho_cat·Cp·ΔTs)':<30} = {delta_Hs:>+16.4e}  [J/m³_bed]",
            f"  {'Q_gs (gas→solido)':<30} = {Q_gs_to_solid:>+16.4e}  [J/m³_bed]",
            f"  {'Q_rxn_solid (hetero rxn)':<30} = {Q_rxn_solid_total:>+16.4e}  [J/m³_bed]",
            f"  {'Q_induction':<30} = {Q_ind_total:>+16.4e}  [J/m³_bed]",
            _line(),
            f"  {'Cierre Hs (ΔHs − pred)':<30} = {resid_Hs:>+16.4e}  "
            f"rel={pct_Hs:+.2f}%  {'★ OK' if pct_Hs < 1.0 else 'aprox'}",
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ENERGÍA PARED — Tw  [J/m³_bed]  (solo si shell_tube)
    # ══════════════════════════════════════════════════════════════════════════
    energy_wall = None
    if shell_tube:
        wall_cfg = params["wall_config"]
        mat      = wall_cfg["material"]
        A_w      = float(wall_cfg["A_w"])

        from src.physics.thermodynamics.solid_props import eval_solid_property
        rho_w_0 = eval_solid_property(mat["rho"], Tw[0])
        cp_w_0  = eval_solid_property(mat["cp"],  Tw[0])
        rho_w_f = eval_solid_property(mat["rho"], Tw[-1])
        cp_w_f  = eval_solid_property(mat["cp"],  Tw[-1])
        # ΔHw = ∫ rho_w * cp_w * (Tw_f − Tw_0) * A_w dz / (Ai * L)
        # normalizado por Ai para obtener [J/m³_bed] coherente con Hg
        Cp_w_m  = 0.5 * (rho_w_0 * cp_w_0 + rho_w_f * cp_w_f)  # (N,)
        delta_Hw = float(np.sum(Cp_w_m * (Tw[-1] - Tw[0]) * A_w * dz)
                         / (Ai * L))

        energy_wall = {"delta_Hw": delta_Hw}
        lines += [
            "", _line("═"),
            "  ENERGÍA PARED — Tw  [J/m³_bed]",
            _line(),
            f"  {'ΔHw (rho_w·cp_w·ΔTw)':<30} = {delta_Hw:>+16.4e}  [J/m³_bed]",
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # Imprimir
    # ══════════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n".join(lines))
        print()

    return {
        "mass_gas":    mass_gas,
        "species_gas": species_gas,
        "energy_gas":  energy_gas,
        "energy_solid": energy_solid,
        "energy_wall":  energy_wall,
    }
