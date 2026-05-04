"""
@module gasifier.config.boundary_c
@brief Build boundary condition configuration for the gasifier.

@details
El modo de operación del gasificador se determina implícitamente a partir de
las condiciones de contorno especificadas, sin necesidad de un parámetro "mode":

    v_gas_in=None  + v_out=0.0   → batch sellado (sistema cerrado, sin flujo)
    v_gas_in=None  + v_out>0     → semibatch (venteo; v_out es la velocidad máxima de alivio)
    v_gas_in≠None  + v_out=None  → CSTR (N=1) o lecho con paso de gas (N>1): v_out por continuidad
    v_gas_in≠None  + v_solid>0  + inlet_mode="prescribed"  → lecho fijo (updraft/downdraft)
    v_gas_in≠None  + v_solid>0  + inlet_mode="computed"    → conveyor (caudal sólido calculado)

En todos los casos, P_out_bar es la presión de referencia del outlet [bar].
"""

import numpy as np


def build_bc_config(
    n_comp: int,
    P_out_bar: float = 1.01325,
    # ── Gas inlet (None = no gas inlet) ──────────────────────────────────────
    v_gas_in=None,          # float or callable(t) → float [m/s]; None = no inlet
    T_gas_in=None,          # float or callable(t) → float [K]; None = no inlet
    y_gas_in=None,          # ndarray(nc,) or callable(t) → ndarray(nc,); None = no inlet
    # ── Gas outlet ────────────────────────────────────────────────────────────
    v_out=None,             # None  = calculate from continuity (v_in × Ctot_in / Ctot_out)
                            # 0.0   = sealed (no outlet flow)
                            # >0    = vent: v_out_actual = max(0,(P−P_out)/P_out)·v_out [m/s]
    # ── Solid (static by default) ─────────────────────────────────────────────
    v_solid: float = 0.0,        # superficial solid velocity [m/s]; 0 = static
    direction=None,               # "updraft" | "downdraft"; required if v_solid > 0
    rho_solid_in=None,            # ndarray(3,) [kg/m³_bed]; required if prescribed solid
    T_solid_in=None,              # float [K]; required if v_solid > 0
    # ── Computed solid inlet (conveyor-like) ──────────────────────────────────
    inlet_mode: str = "prescribed",   # "prescribed" | "computed"
    inlet_method: str = "explicit",   # "explicit" | "implicit" (only for "computed")
    rho_solid_fresh_total=None,        # float [kg/m³_bed]; initial density (for "computed")
    mc_wb=None,                        # float [-]; moisture content wet basis (for "computed")
) -> dict:
    """
    Build and validate the boundary condition configuration dict.

    Parameters
    ----------
    n_comp : int
        Number of gas species (9 for the gasifier).
    P_out_bar : float
        Gas outlet pressure reference [bar].
    v_gas_in : float, callable(t) → float, or None
        Superficial gas velocity at inlet [m/s]. None = no gas inlet.
    T_gas_in : float, callable(t) → float, or None
        Gas temperature at inlet [K]. Required if v_gas_in is not None.
    y_gas_in : ndarray(nc,), callable(t) → ndarray(nc,), or None
        Gas molar fractions at inlet [-]. Required if v_gas_in is not None.
    v_out : float >= 0 or None
        Outlet strategy:
        None  → v_out derived from molar continuity (v_out = v_in·Ctot_in/Ctot_out).
                When v_gas_in is None this gives v_out = 0 automatically.
        0.0   → sealed outlet: v_out = 0 always (batch pyrolysis).
        > 0   → vent mode: v_out = max(0, (P−P_out)/P_out) · v_out [m/s].
                v_out is the maximum vent velocity (valve fully open at P = 2·P_out).
    v_solid : float
        Superficial solid velocity [m/s]. 0 = static solid.
    direction : {"updraft", "downdraft"} or None
        Direction of solid movement. Required if v_solid > 0.
    rho_solid_in : ndarray(3,) or None
        Solid bulk densities [biomass, char, moisture] at inlet [kg/m³_bed].
    T_solid_in : float or None
        Solid temperature at inlet [K]. Required if v_solid > 0.
    inlet_mode : {"prescribed", "computed"}
        "prescribed" → rho_solid_in supplied explicitly.
        "computed"   → rho_solid_in computed in the RHS from the solid mass balance.
    inlet_method : {"explicit", "implicit"}
        Only for inlet_mode="computed".
    rho_solid_fresh_total : float or None
        Total bulk density of fresh solid feed [kg/m³_bed]. Required for "computed".
    mc_wb : float or None
        Moisture content of fresh feed in wet basis [-]. Required for "computed".

    Returns
    -------
    dict with keys:
        P_out_bar, v_gas_in, T_gas_in, y_gas_in,
        v_out,
        v_solid, direction, rho_solid_in, T_solid_in,
        inlet_mode, inlet_method, rho_solid_fresh_total, mc_wb
    """
    if P_out_bar <= 0.0:
        raise ValueError(f"P_out_bar must be > 0, got {P_out_bar}")

    # ── Validación de v_out ───────────────────────────────────────────────────
    if v_out is not None:
        v_out_f = float(v_out)
        if v_out_f < 0.0:
            raise ValueError(f"v_out must be >= 0 or None, got {v_out_f}")
    else:
        v_out_f = None

    # ── Validación del inlet de gas ───────────────────────────────────────────
    if v_gas_in is not None:
        if T_gas_in is None:
            raise ValueError("T_gas_in is required when v_gas_in is not None")
        if y_gas_in is None:
            raise ValueError("y_gas_in is required when v_gas_in is not None")
        if not callable(v_gas_in) and float(v_gas_in) < 0.0:
            raise ValueError(f"v_gas_in must be >= 0, got {v_gas_in}")
        if not callable(T_gas_in) and float(T_gas_in) <= 0.0:
            raise ValueError(f"T_gas_in must be > 0, got {T_gas_in}")
        if not callable(y_gas_in):
            y_arr = np.asarray(y_gas_in, dtype=float).reshape(-1)
            if y_arr.shape != (n_comp,):
                raise ValueError(
                    f"y_gas_in must have length n_comp={n_comp}, got {y_arr.shape}"
                )
            if abs(float(np.sum(y_arr)) - 1.0) > 1.0e-4:
                raise ValueError(
                    f"y_gas_in must sum to 1, got {float(np.sum(y_arr)):.6f}"
                )
            y_gas_in = y_arr

    # ── Validación del sólido ─────────────────────────────────────────────────
    v_solid_f = float(v_solid)
    if v_solid_f < 0.0:
        raise ValueError(f"v_solid must be >= 0, got {v_solid_f}")

    direction_str = None
    if v_solid_f > 0.0:
        if direction is None:
            raise ValueError(
                "direction ('updraft' or 'downdraft') is required when v_solid > 0"
            )
        direction_str = str(direction).strip().lower()
        if direction_str not in {"updraft", "downdraft"}:
            raise ValueError(
                f"direction must be 'updraft' or 'downdraft', got '{direction_str}'"
            )
        if T_solid_in is None:
            raise ValueError("T_solid_in is required when v_solid > 0")

    inlet_mode_str = str(inlet_mode).strip().lower()
    if inlet_mode_str not in {"prescribed", "computed"}:
        raise ValueError(
            f"inlet_mode must be 'prescribed' or 'computed', got '{inlet_mode_str}'"
        )

    # Sólido prescrito: validar rho_solid_in
    if v_solid_f > 0.0 and inlet_mode_str == "prescribed":
        if rho_solid_in is None:
            raise ValueError(
                "rho_solid_in is required when v_solid > 0 and inlet_mode='prescribed'"
            )
        rho_s = np.asarray(rho_solid_in, dtype=float).reshape(-1)
        if rho_s.shape != (3,):
            raise ValueError(f"rho_solid_in must have shape (3,), got {rho_s.shape}")
        if np.any(rho_s < 0.0):
            raise ValueError("rho_solid_in values must be >= 0")
        rho_solid_in = rho_s

    # Sólido calculado (conveyor): validar parámetros adicionales
    if inlet_mode_str == "computed":
        if v_solid_f <= 0.0:
            raise ValueError("v_solid must be > 0 for inlet_mode='computed'")
        if rho_solid_fresh_total is None:
            raise ValueError(
                "rho_solid_fresh_total is required for inlet_mode='computed'"
            )
        if mc_wb is None:
            raise ValueError("mc_wb is required for inlet_mode='computed'")
        if float(rho_solid_fresh_total) <= 0.0:
            raise ValueError(
                f"rho_solid_fresh_total must be > 0, got {rho_solid_fresh_total}"
            )
        mc_f = float(mc_wb)
        if not (0.0 <= mc_f < 1.0):
            raise ValueError(f"mc_wb must be in [0, 1), got {mc_f}")
        mc_wb = mc_f
        rho_solid_fresh_total = float(rho_solid_fresh_total)

    inlet_method_str = str(inlet_method).strip().lower()
    if inlet_method_str not in {"explicit", "implicit"}:
        raise ValueError(
            f"inlet_method must be 'explicit' or 'implicit', got '{inlet_method_str}'"
        )

    return {
        "P_out_bar":             float(P_out_bar),
        "v_gas_in":              v_gas_in,
        "T_gas_in":              T_gas_in,
        "y_gas_in":              y_gas_in,
        "v_out":                 v_out_f,
        "v_solid":               v_solid_f,
        "direction":             direction_str,
        "rho_solid_in":          rho_solid_in,
        "T_solid_in":            float(T_solid_in) if T_solid_in is not None else None,
        "inlet_mode":            inlet_mode_str,
        "inlet_method":          inlet_method_str,
        "rho_solid_fresh_total": rho_solid_fresh_total,
        "mc_wb":                 mc_wb,
    }
