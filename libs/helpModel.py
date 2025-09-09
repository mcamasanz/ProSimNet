# -*- coding: utf-8 -*-
"""
guia_transporte_markdown.py
Guía universal (en español) para interpretar números adimensionales
y coeficientes de transporte (masa, cantidad de movimiento y energía),
generando informes en formato Markdown (amigable para Jupyter).

Incluye validaciones frente a correlaciones clásicas:
- Dittus–Boelter (tubo turbulento)
- Gnielinski (tubo turbulento)
- Ranz–Marshall (esfera en flujo externo)
- Analogía de Chilton–Colburn (masa ↔ calor)

Autor: tú + ChatGPT (GPT-5 Thinking)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np

Number = Union[float, int]
ArrayLike = Union[Number, List[Number], np.ndarray]

# ============
# Utilidades
# ============

def _to_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    return np.asarray([x], dtype=float)

def _fmt_value(x: ArrayLike, fmt: str = "{:.4g}") -> str:
    a = _to_array(x)
    if a.size == 1:
        return fmt.format(float(a[0]))
    return f"mean={fmt.format(a.mean())}, min={fmt.format(a.min())}, max={fmt.format(a.max())}"

def _md_escape(s: str) -> str:
    # Evita que '_' o '|' rompan tablas en markdown
    return s.replace("_", "\\_").replace("|", "\\|")

# =======================================
# Definición de magnitudes
# =======================================

@dataclass
class Magnitud:
    nombre: str
    simbolo_md: str      # p.ej. r"$\mathrm{Re}$"
    tipo: str            # "adimensional" | "coeficiente"
    formula_latex: str   # sin f-strings; se renderiza en Markdown con $$ ... $$
    unidades_md: str     # p.ej. "[-]" o r"$\mathrm{W\,m^{-2}K^{-1}}$"
    variables: List[Tuple[str, str, str]]  # [(símbolo, desc corta, unidades)]
    descripcion: str     # ¿Qué es / qué describe?
    interpreta: Callable[[np.ndarray, Dict], str]
    recomendaciones: Callable[[np.ndarray, Dict], str]

# ============================
# Clasificadores y utilidades
# ============================

def clasifica_re(Re: np.ndarray, ctx: Dict) -> str:
    geom = ctx.get("geometria", "tubo_interno")
    if geom == "tubo_interno":
        c = []
        for r in Re:
            if r < 2100:
                c.append(f"Re={r:.0f}: laminar (flujo interno). Perfil parabólico; Nu≈3.66–4.36 si totalmente desarrollado.")
            elif r < 4000:
                c.append(f"Re={r:.0f}: transición. Resultado sensible a perturbaciones/entrada.")
            else:
                c.append(f"Re={r:.0f}: turbulento. Mezcla intensa; válidas Dittus–Boelter/Gnielinski.")
        return "\n".join(f"- {line}" for line in c)
    elif geom == "esfera_externo":
        return "- Flujo externo sobre partícula: usar Ranz–Marshall si $\\mathrm{Re}\\lesssim 2\\times10^5$."
    elif geom == "lecho_empacado":
        return "- Lecho empacado: usar $\\mathrm{Re}_p$ (Wakao & Funazkri)."
    else:
        return "- Geometría no especificada: interpreta Re según el caso."

def recs_re(Re: np.ndarray, ctx: Dict) -> str:
    geom = ctx.get("geometria", "tubo_interno")
    if geom == "tubo_interno":
        if np.all(Re < 2100):
            return "- Modelo: incluir difusión viscosa; en 1D usar coeficientes laminares; evita correlaciones turbulentas."
        if np.any((Re >= 2100) & (Re < 4000)):
            return "- Modelo: transición; Gnielinski con cautela o RANS/LES si es crítico."
        return "- Modelo: turbulento; Dittus–Boelter/Gnielinski. Posible ignorar dispersión axial si $\\mathrm{Pe}\\gg 1$."
    return "- Elegir correlación según geometría (Ranz–Marshall, Wakao & Funazkri, etc.)."

def clasifica_pe(Pe: np.ndarray, ctx: Dict) -> str:
    c = []
    for p in Pe:
        if p < 0.1:
            c.append(f"Pe={p:.3g}: fuertemente difusivo; advección despreciable.")
        elif p < 2:
            c.append(f"Pe={p:.3g}: difusión relevante; diferencias centradas estables.")
        elif p < 10:
            c.append(f"Pe={p:.3g}: convección moderada; usar upwind/TVD/WENO.")
        else:
            c.append(f"Pe={p:.3g}: fuertemente convectivo; upwind/WENO; dispersión axial puede ignorarse.")
    return "\n".join(f"- {line}" for line in c)

def recs_pe(Pe: np.ndarray, ctx: Dict) -> str:
    if np.all(Pe < 2):
        return "- Discretización: centradas. Modelo: incluir dispersión/difusión axial."
    if np.any((Pe >= 2) & (Pe < 10)):
        return "- Discretización: upwind/TVD. Modelo: considerar dispersión axial si hay tortuosidad/heterogeneidad."
    return "- Discretización: upwind/WENO. Modelo: la advección domina."

def clasifica_bi(Bi: np.ndarray, ctx: Dict) -> str:
    c = []
    for b in Bi:
        if b < 0.1:
            c.append(f"Bi={b:.3g}: película poco resistiva; sólido casi lumped.")
        elif b < 10:
            c.append(f"Bi={b:.3g}: resistencias comparables; gradientes internos y en película.")
        else:
            c.append(f"Bi={b:.3g}: resistencia interna domina; fuertes gradientes en el sólido.")
    return "\n".join(f"- {line}" for line in c)

def recs_bi(Bi: np.ndarray, ctx: Dict) -> str:
    if np.all(Bi < 0.1):
        return "- Modelo: capacidad concentrada (lumped) aceptable."
    if np.any(Bi > 10):
        return "- Modelo: resolver difusión interna (transitorio/estacionario)."
    return "- Modelo: resistencias en serie (película + interno)."

def clasifica_da(Da: np.ndarray, ctx: Dict) -> str:
    c = []
    for d in Da:
        if d < 0.1:
            c.append(f"Da={d:.3g}: control por convección (poca conversión).")
        elif d < 10:
            c.append(f"Da={d:.3g}: régimen mixto (tiempos comparables).")
        else:
            c.append(f"Da={d:.3g}: control difusivo/kinético.")
    return "\n".join(f"- {line}" for line in c)

def recs_da(Da: np.ndarray, ctx: Dict) -> str:
    if np.all(Da < 0.1):
        return "- Mejorar convección externa (↑u, ↑$k_c$)."
    if np.any(Da > 10):
        return "- ↑$D_\\text{intra}$/↓tortuosidad o ↓tamaño de partícula."
    return "- Ambos mecanismos relevantes; calibrar con datos."

def clasifica_sh_nu(X: np.ndarray, ctx: Dict, etiqueta: str) -> str:
    c = []
    for v in X:
        if v <= 2.5:
            c.append(f"{etiqueta}≈{v:.2f}: difusión/conducción dominante (≈2).")
        elif v < 10:
            c.append(f"{etiqueta}={v:.2f}: convección moderada.")
        else:
            c.append(f"{etiqueta}={v:.2f}: convección intensa.")
    return "\n".join(f"- {line}" for line in c)

def recs_sh_nu(X: np.ndarray, ctx: Dict, etiqueta: str) -> str:
    if np.all(X <= 2.5):
        return "- Control difusivo/conductivo; película resistiva."
    if np.any(X >= 10):
        return "- Película fina; $h$ o $k_c$ elevados; posible despreciar difusión axial."
    return "- Régimen mixto; usar correlaciones adecuadas."

def clasifica_sc_pr(S: np.ndarray, ctx: Dict, etiqueta: str) -> str:
    c = []
    for s in S:
        if s < 0.7:
            c.append(f"{etiqueta}={s:.2f}: (gases ligeros) difusión relativamente rápida.")
        elif s <= 2:
            c.append(f"{etiqueta}={s:.2f}: rango típico de gases/líquidos poco viscosos.")
        else:
            c.append(f"{etiqueta}={s:.2f}: (líquidos viscosos) difusión molecular lenta.")
    return "\n".join(f"- {line}" for line in c)

def recs_sc_pr(S: np.ndarray, ctx: Dict, etiqueta: str) -> str:
    if np.any(S > 2):
        return "- Refinar gradientes junto a pared/partícula; sensibilidad al mallado."
    return "- Analogías (Chilton–Colburn) con precaución si hay propiedades variables."

# =========================
# Correlaciones (validación)
# =========================

def nu_dittus_boelter(Re: float, Pr: float, calentando: bool = True) -> float:
    n = 0.4 if calentando else 0.3
    return 0.023 * (Re ** 0.8) * (Pr ** n)

def f_gnielinski(Re: float) -> float:
    return (0.79 * math.log(Re) - 1.64) ** (-2)

def nu_gnielinski(Re: float, Pr: float) -> float:
    f = f_gnielinski(Re)
    return (f / 8.0) * (Re - 1000.0) * Pr / (1 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2/3) - 1))

def ranz_marshall(Re: float, Pr_or_Sc: float) -> float:
    return 2.0 + 0.6 * (Re ** 0.5) * (Pr_or_Sc ** (1.0 / 3.0))

def chilton_colburn_sh(Re: float, Sc: float, calentando: bool = True) -> float:
    n = 1/3 if calentando else 0.4
    return 0.023 * (Re ** 0.8) * (Sc ** n)

# =======================
# Biblioteca de magnitudes
# =======================

def build_magnitudes() -> Dict[str, Magnitud]:
    mags: Dict[str, Magnitud] = {}

    mags["Re"] = Magnitud(
        nombre="Número de Reynolds",
        simbolo_md=r"$\mathrm{Re}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Re}=\dfrac{\rho\,u\,L}{\mu}",
        unidades_md="[-]",
        variables=[(r"$\rho$", "densidad", r"$\mathrm{kg\,m^{-3}}$"),
                   (r"$u$", "velocidad", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$"),
                   (r"$\mu$", "visc. dinámica", r"$\mathrm{Pa\,s}$")],
        descripcion="Relaciona inercia y fricción viscosa; clasifica el régimen (laminar/transición/turbulento).",
        interpreta=lambda v, ctx: clasifica_re(v, ctx),
        recomendaciones=lambda v, ctx: recs_re(v, ctx),
    )

    mags["Pr"] = Magnitud(
        nombre="Número de Prandtl",
        simbolo_md=r"$\mathrm{Pr}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Pr}=\dfrac{\mu\,c_p}{k}=\dfrac{\nu}{\alpha}",
        unidades_md="[-]",
        variables=[(r"$\mu$", "visc. dinámica", r"$\mathrm{Pa\,s}$"),
                   (r"$c_p$", "calor específico", r"$\mathrm{J\,kg^{-1}K^{-1}}$"),
                   (r"$k$", "conductividad fluido", r"$\mathrm{W\,m^{-1}K^{-1}}$"),
                   (r"$\nu$", "visc. cinemática", r"$\mathrm{m^2\,s^{-1}}$"),
                   (r"$\alpha$", "difusividad térmica", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Compara difusión de cantidad de movimiento con la térmica; gobierna la capa límite térmica.",
        interpreta=lambda v, ctx: clasifica_sc_pr(v, ctx, "Pr"),
        recomendaciones=lambda v, ctx: recs_sc_pr(v, ctx, "Pr"),
    )

    mags["Sc"] = Magnitud(
        nombre="Número de Schmidt",
        simbolo_md=r"$\mathrm{Sc}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Sc}=\dfrac{\mu}{\rho\,D_{AB}}=\dfrac{\nu}{D_{AB}}",
        unidades_md="[-]",
        variables=[(r"$\mu$", "visc. dinámica", r"$\mathrm{Pa\,s}$"),
                   (r"$\rho$", "densidad", r"$\mathrm{kg\,m^{-3}}$"),
                   (r"$D_{AB}$", "difusividad mol.", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Compara difusión de cantidad de movimiento con la de masa; gobierna la capa límite de especie.",
        interpreta=lambda v, ctx: clasifica_sc_pr(v, ctx, "Sc"),
        recomendaciones=lambda v, ctx: recs_sc_pr(v, ctx, "Sc"),
    )

    mags["Pe"] = Magnitud(
        nombre="Número de Péclet (genérico)",
        simbolo_md=r"$\mathrm{Pe}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Pe}=\dfrac{u\,L}{\mathcal{D}} \;(=\mathrm{Re\,Pr}\ \text{o}\ \mathrm{Re\,Sc})",
        unidades_md="[-]",
        variables=[(r"$u$", "velocidad", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$"),
                   (r"$\mathcal{D}$", "difusividad térmica/masa", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Mide la relación advección/difusión; clave para escoger esquema numérico.",
        interpreta=lambda v, ctx: clasifica_pe(v, ctx),
        recomendaciones=lambda v, ctx: recs_pe(v, ctx),
    )

    mags["Nu"] = Magnitud(
        nombre="Número de Nusselt",
        simbolo_md=r"$\mathrm{Nu}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Nu}=\dfrac{h\,L}{k}",
        unidades_md="[-]",
        variables=[(r"$h$", "coef. convectivo", r"$\mathrm{W\,m^{-2}K^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$"),
                   (r"$k$", "cond. térmica fluido", r"$\mathrm{W\,m^{-1}K^{-1}}$")],
        descripcion="Relación convección/conducción en una capa límite térmica.",
        interpreta=lambda v, ctx: clasifica_sh_nu(v, ctx, "Nu"),
        recomendaciones=lambda v, ctx: recs_sh_nu(v, ctx, "Nu"),
    )

    mags["Sh"] = Magnitud(
        nombre="Número de Sherwood",
        simbolo_md=r"$\mathrm{Sh}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Sh}=\dfrac{k_c\,L}{D_{AB}}",
        unidades_md="[-]",
        variables=[(r"$k_c$", "coef. convectivo masa", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$"),
                   (r"$D_{AB}$", "difusividad mol.", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Análogo másico de Nusselt: convección vs difusión de especie.",
        interpreta=lambda v, ctx: clasifica_sh_nu(v, ctx, "Sh"),
        recomendaciones=lambda v, ctx: recs_sh_nu(v, ctx, "Sh"),
    )

    mags["Bi"] = Magnitud(
        nombre="Número de Biot (térmico/masivo)",
        simbolo_md=r"$\mathrm{Bi}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Bi}=\dfrac{h\,L_c}{k_s}\quad(\text{o } \mathrm{Bi}_m=\dfrac{k_c\,L_c}{D_{\mathrm{intra}}})",
        unidades_md="[-]",
        variables=[(r"$h$", "coef. convectivo calor", r"$\mathrm{W\,m^{-2}K^{-1}}$"),
                   (r"$k_c$", "coef. convectivo masa", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$L_c$", "long. característica sólido", r"$\mathrm{m}$"),
                   (r"$k_s$", "cond. térmica sólido", r"$\mathrm{W\,m^{-1}K^{-1}}$"),
                   (r"$D_{\mathrm{intra}}$", "difusividad intrapartícula", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Compara la resistencia de película con la interna del sólido/partícula.",
        interpreta=lambda v, ctx: clasifica_bi(v, ctx),
        recomendaciones=lambda v, ctx: recs_bi(v, ctx),
    )

    mags["Da_ext"] = Magnitud(
        nombre="Número de Damköhler externo",
        simbolo_md=r"$\mathrm{Da}_{\mathrm{ext}}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Da}_{\mathrm{ext}}=\dfrac{k_c\,a_s\,L}{u}",
        unidades_md="[-]",
        variables=[(r"$k_c$", "coef. conv. masa", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$a_s$", "área específica", r"$\mathrm{m^2\,m^{-3}}$"),
                   (r"$L$", "longitud lecho/columna", r"$\mathrm{m}$"),
                   (r"$u$", "velocidad superficial", r"$\mathrm{m\,s^{-1}}$")],
        descripcion="Relación entre tiempo convectivo y de transferencia externa de masa.",
        interpreta=lambda v, ctx: clasifica_da(v, ctx),
        recomendaciones=lambda v, ctx: recs_da(v, ctx),
    )

    mags["Da_ldf"] = Magnitud(
        nombre="Número de Damköhler LDF",
        simbolo_md=r"$\mathrm{Da}_{\mathrm{LDF}}$",
        tipo="adimensional",
        formula_latex=r"\mathrm{Da}_{\mathrm{LDF}}=\dfrac{k_{\mathrm{LDF}}\,L}{u/\varepsilon}",
        unidades_md="[-]",
        variables=[(r"$k_{\mathrm{LDF}}$", "coef. LDF interno", r"$\mathrm{s^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$"),
                   (r"$u/\varepsilon$", "vel. intersticial", r"$\mathrm{m\,s^{-1}}$")],
        descripcion="Compara la cinética intrapartícula (LDF) con la convección a escala de celda.",
        interpreta=lambda v, ctx: clasifica_da(v, ctx),
        recomendaciones=lambda v, ctx: recs_da(v, ctx),
    )

    mags["h"] = Magnitud(
        nombre="Coeficiente convectivo de calor",
        simbolo_md=r"$h$",
        tipo="coeficiente",
        formula_latex=r"h=\dfrac{\mathrm{Nu}\,k}{L}",
        unidades_md=r"$\mathrm{W\,m^{-2}K^{-1}}$",
        variables=[(r"$\mathrm{Nu}$", "Nusselt", "-"),
                   (r"$k$", "cond. térmica fluido", r"$\mathrm{W\,m^{-1}K^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$")],
        descripcion="Intensidad de convección térmica en la interfase fluido-sólido.",
        interpreta=lambda v, ctx: "Valores altos ⇒ película térmica delgada; bajos ⇒ control por conducción en película.",
        recomendaciones=lambda v, ctx: "Si $Bi\\ll1$: lumped aceptable; si $Bi\\gg1$: resolver difusión interna.",
    )

    mags["k_c"] = Magnitud(
        nombre="Coeficiente convectivo de masa",
        simbolo_md=r"$k_c$",
        tipo="coeficiente",
        formula_latex=r"k_c=\dfrac{\mathrm{Sh}\,D_{AB}}{L}",
        unidades_md=r"$\mathrm{m\,s^{-1}}$",
        variables=[(r"$\mathrm{Sh}$", "Sherwood", "-"),
                   (r"$D_{AB}$", "difusividad mol.", r"$\mathrm{m^2\,s^{-1}}$"),
                   (r"$L$", "longitud car.", r"$\mathrm{m}$")],
        descripcion="Intensidad de convección de especie a través de la película externa.",
        interpreta=lambda v, ctx: "Alto $k_c$ ⇒ película de masa delgada; menor control difusivo externo.",
        recomendaciones=lambda v, ctx: "↑u/turbulencia/rugosidad → ↑$k_c$; en $Pe\\gg1$ suele no ser limitante.",
    )

    mags["k_LDF"] = Magnitud(
        nombre="Coeficiente LDF intrapartícula",
        simbolo_md=r"$k_{\mathrm{LDF}}$",
        tipo="coeficiente",
        formula_latex=r"\dfrac{\mathrm{d}q}{\mathrm{d}t} = k_{\mathrm{LDF}}(q^*-q)",
        unidades_md=r"$\mathrm{s^{-1}}$",
        variables=[(r"$q$", "carga en partícula", r"$\mathrm{mol\,kg^{-1}}$"),
                   (r"$q^*$", "equilibrio (isoterma)", r"$\mathrm{mol\,kg^{-1}}$")],
        descripcion="Parámetro efectivo para difusión/film interno (modelo LDF).",
        interpreta=lambda v, ctx: "Mayor $k_{LDF}$ ⇒ respuesta interna rápida (menor resistencia).",
        recomendaciones=lambda v, ctx: "Si $Da_{LDF}\\gg1$ control interno; reducir tamaño partícula o ↑T.",
    )

    mags["D_ij"] = Magnitud(
        nombre="Difusividad binaria (gas)",
        simbolo_md=r"$D_{ij}$",
        tipo="coeficiente",
        formula_latex=r"\text{(depende de $T$, $P$, $x_i$)}",
        unidades_md=r"$\mathrm{m^2\,s^{-1}}$",
        variables=[(r"$T$", "temperatura", "K"),
                   (r"$P$", "presión", "Pa"),
                   (r"$x_i$", "fracción molar", "-")],
        descripcion="Difusión molecular de una especie en otra (mezclas diluidas).",
        interpreta=lambda v, ctx: "Gases típicos: $10^{-5}$–$10^{-4}$; líquidos: $10^{-9}$–$10^{-10}$ (m²/s).",
        recomendaciones=lambda v, ctx: "↑T o ↓P ⇒ ↑$D_{ij}$; considerar Knudsen si poros pequeños.",
    )

    mags["D_kn"] = Magnitud(
        nombre="Difusividad de Knudsen",
        simbolo_md=r"$D_{\mathrm{kn}}$",
        tipo="coeficiente",
        formula_latex=r"D_{\mathrm{kn}}\sim \dfrac{2}{3}\,r_p\,\sqrt{\dfrac{8RT}{\pi M}}",
        unidades_md=r"$\mathrm{m^2\,s^{-1}}$",
        variables=[(r"$r_p$", "radio de poro", "m"),
                   (r"$T$", "temperatura", "K"),
                   (r"$M$", "masa molar", r"$\mathrm{kg\,mol^{-1}}$")],
        descripcion="Régimen de colisiones con la pared cuando la longitud libre media ≳ radio de poro.",
        interpreta=lambda v, ctx: "Relevante si $Kn$ grande (poros meso/micros).",
        recomendaciones=lambda v, ctx: "Combinar con $D_m$ (Bosánquet) para $D_{eff}$.",
    )

    mags["D_eff"] = Magnitud(
        nombre="Difusividad efectiva en poro",
        simbolo_md=r"$D_{\mathrm{eff}}$",
        tipo="coeficiente",
        formula_latex=r"D_{\mathrm{eff}}=\dfrac{\varepsilon}{\tau}\left(\dfrac{1}{D_m}+\dfrac{1}{D_{\mathrm{kn}}}\right)^{-1}",
        unidades_md=r"$\mathrm{m^2\,s^{-1}}$",
        variables=[(r"$\varepsilon$", "porosidad", "-"),
                   (r"$\tau$", "tortuosidad", "-"),
                   (r"$D_m$", "difusividad mol.", r"$\mathrm{m^2\,s^{-1}}$"),
                   (r"$D_{\mathrm{kn}}$", "dif. Knudsen", r"$\mathrm{m^2\,s^{-1}}$")],
        descripcion="Difusividad aparente en medios porosos: combina porosidad, tortuosidad y regímenes difusivos.",
        interpreta=lambda v, ctx: "Siempre menor que $D_m$; limita el transporte interno.",
        recomendaciones=lambda v, ctx: "↓tortuosidad o ↑porosidad para aliviar resistencia interna.",
    )

    mags["D_z"] = Magnitud(
        nombre="Dispersión axial efectiva",
        simbolo_md=r"$D_z$",
        tipo="coeficiente",
        formula_latex=r"\text{Taylor–Aris / Lechos empacados (varias correlaciones)}",
        unidades_md=r"$\mathrm{m^2\,s^{-1}}$",
        variables=[(r"$u$", "velocidad", r"$\mathrm{m\,s^{-1}}$"),
                   (r"$\alpha_L$", "long. de dispersión", r"$\mathrm{m}$")],
        descripcion="Mezcla axial debida a perfiles de velocidad y difusión; clave en columnas y lechos.",
        interpreta=lambda v, ctx: "Relaciona con $Pe_{ax}=uL/D_z$: pequeño ⇒ mezcla axial alta.",
        recomendaciones=lambda v, ctx: "Si $Pe_{ax}<2$, centradas; si $\\gg 2$, upwind/WENO.",
    )

    return mags

# ========================
# Informe Markdown
# ========================

def render_magnitud_section_md(m: Magnitud, valor: ArrayLike, contexto: Dict) -> str:
    v_arr = _to_array(valor)
    valor_md = _fmt_value(v_arr)
    # Tabla de variables: solo keywords breves (no frases largas)
    header = "| Símbolo | Clave | Unidades |\n|---|---|---|"
    rows = "\n".join([f"| {sym} | { _md_escape(desc) } | {units} |" for sym, desc, units in m.variables])

    s  = f"\n### { _md_escape(m.nombre) } ({m.simbolo_md})\n\n"
    s += f"**¿Qué es?** {m.descripcion}\n\n"
    s += f"**Fórmula**\n\n$$ {m.formula_latex} $$\n\n"
    s += f"**Tipo:** {m.tipo} &nbsp;&nbsp; **Unidades:** {m.unidades_md} &nbsp;&nbsp; **Valor(es):** `{valor_md}`\n\n"
    s += f"**Variables**\n{header}\n{rows}\n\n"
    s += f"**Interpretación según el valor**\n{m.interpreta(v_arr, contexto)}\n\n"
    s += f"**Recomendaciones de modelado / numéricas**\n{m.recomendaciones(v_arr, contexto)}\n\n"
    s += "---\n"
    return s

def render_resumen_md(magnitudes: Dict[str, Magnitud],
                      valores: Dict[str, ArrayLike]) -> str:
    header = "| Magnitud | Símbolo | Valor | Unidades |\n|---|---|---|---|"
    rows = []
    for key in valores:
        if key not in magnitudes: 
            continue
        m = magnitudes[key]
        rows.append(f"| {_md_escape(m.nombre)} | {m.simbolo_md} | `{_fmt_value(valores[key])}` | {m.unidades_md} |")
    return "## Tabla resumen\n\n" + header + "\n" + "\n".join(rows) + "\n"

def generar_informe_markdown(titulo: str,
                             magnitudes: Dict[str, Magnitud],
                             valores: Dict[str, ArrayLike],
                             contexto: Dict,
                             validaciones: Optional[List[str]] = None) -> str:
    md = [f"# { _md_escape(titulo) }\n",
          "> Guía automática de interpretación de números adimensionales y coeficientes de transporte.\n",
          f"**Contexto**: {_md_escape(contexto.get('descripcion','(sin descripción)'))}\n\n"]

    orden = ["Re", "Pr", "Sc", "Pe", "Nu", "Sh", "Bi", "Da_ext", "Da_ldf",
             "h", "k_c", "k_LDF", "D_ij", "D_kn", "D_eff", "D_z"]
    for key in orden:
        if key in valores and key in magnitudes:
            md.append(render_magnitud_section_md(magnitudes[key], valores[key], contexto))

    if validaciones:
        md.append("## Validaciones con correlaciones\n")
        for line in validaciones:
            md.append(f"- {line}")
        md.append("\n")

    md.append(render_resumen_md(magnitudes, valores))
    return "".join(md)

# ======================
# Validadores (pruebas)
# ======================

def validar_termico_tubo(Re: float, Pr: float, Nu_reportado: float) -> str:
    nu_db = nu_dittus_boelter(Re, Pr, calentando=True)
    try:
        nu_gn = nu_gnielinski(Re, Pr)
    except Exception:
        nu_gn = float('nan')
    err_db = abs(Nu_reportado - nu_db) / max(nu_db, 1e-12) * 100.0
    err_gn = abs(Nu_reportado - nu_gn) / max(nu_gn, 1e-12) * 100.0 if math.isfinite(nu_gn) and nu_gn > 0 else float('nan')
    return (f"**Tubo (turbulento)**: Nu_rep={Nu_reportado:.2f}, "
            f"Nu_DB={nu_db:.2f} (error {err_db:.1f}%), "
            f"Nu_Gnielinski={nu_gn:.2f} (error {err_gn:.1f}%).")

def validar_masico(Re: float, Sc: float, Sh_reportado: float, geometria: str) -> str:
    if geometria == "esfera_externo":
        sh_rm = ranz_marshall(Re, Sc)
        err = abs(Sh_reportado - sh_rm) / max(sh_rm, 1e-12) * 100.0
        return (f"**Esfera externo**: Sh_rep={Sh_reportado:.1f}, "
                f"Sh_Ranz–Marshall={sh_rm:.1f} (error {err:.1f}%).")
    else:
        sh_cc = chilton_colburn_sh(Re, Sc)
        err = abs(Sh_reportado - sh_cc) / max(sh_cc, 1e-12) * 100.0
        return (f"**Tubo (analogía C–C)**: Sh_rep={Sh_reportado:.1f}, "
                f"Sh_CC={sh_cc:.1f} (error {err:.1f}%).")

# ======================
# Ejemplos A, B, C
# ======================

def ejemplo_A():
    # Gas en tubo turbulento (propiedades tipo gases)
    Re = 15000.0; Pr = 0.77; Sc = 0.60
    D = 0.02; k = 0.018; D_AB = 1.0e-5
    Nu_rep = nu_dittus_boelter(Re, Pr, True)
    h = Nu_rep * k / D
    Sh_rep = chilton_colburn_sh(Re, Sc)
    k_c = Sh_rep * D_AB / D
    Bi = h * (D/2) / 3.0
    Da_ext = 5e5; Da_ldf = 2.0e4; Pe = Re * Pr
    valid = [
        validar_termico_tubo(Re, Pr, Nu_rep),
        validar_masico(Re, Sc, Sh_rep, "tubo_interno")
    ]
    valores = {
        "Re": Re, "Pr": Pr, "Sc": Sc, "Pe": Pe,
        "Nu": Nu_rep, "Sh": Sh_rep, "Bi": Bi,
        "Da_ext": Da_ext, "Da_ldf": Da_ldf,
        "h": h, "k_c": k_c, "k_LDF": 200.0,
        "D_ij": 1.2e-5, "D_kn": 3e-5, "D_eff": 8e-6, "D_z": 1e-3
    }
    ctx = {"geometria": "tubo_interno",
           "descripcion": "Flujo interno turbulento; convección domina; upwind/WENO"}
    titulo = "Caso A — Régimen turbulento (tubo interno), Pe ≫ 1"
    return titulo, valores, ctx, valid

def ejemplo_B():
    # Líquido viscoso en régimen laminar; transporte difusivo relevante
    Re = 500.0; Pr = 7.0; Sc = 1200.0
    D = 0.01; k = 0.6; D_AB = 1.0e-9
    Nu_rep = 3.66
    h = Nu_rep * k / D
    Sh_rep = 3.66
    k_c = Sh_rep * D_AB / D
    Bi = h * (D/2) / 20.0
    Da_ext = 0.5; Da_ldf = 0.2; Pe = 1.2  # Pe de celda
    valid = [
        validar_termico_tubo(Re, Pr, Nu_rep),
        validar_masico(Re, Sc, Sh_rep, "tubo_interno")
    ]
    valores = {
        "Re": Re, "Pr": Pr, "Sc": Sc, "Pe": Pe,
        "Nu": Nu_rep, "Sh": Sh_rep, "Bi": Bi,
        "Da_ext": Da_ext, "Da_ldf": Da_ldf,
        "h": h, "k_c": k_c, "k_LDF": 0.05,
        "D_ij": 1.0e-9, "D_kn": 0.0, "D_eff": 5e-10, "D_z": 5e-6
    }
    ctx = {"geometria": "tubo_interno",
           "descripcion": "Flujo laminar-difusivo; centradas; controlar difusión molecular"}
    titulo = "Caso B — Régimen laminar-difusivo (tubo interno), PeΔ < 2"
    return titulo, valores, ctx, valid

def ejemplo_C():
    # Transición/intermedio; convección y difusión comparables
    Re = 2200.0; Pr = 1.0; Sc = 1.0
    D = 0.015; k = 0.03; D_AB = 2.0e-5
    Nu_rep = 10.0
    h = Nu_rep * k / D
    Sh_rep = 15.0
    k_c = Sh_rep * D_AB / D
    Bi = h * (D/2) / 1.5
    Da_ext = 1.0; Da_ldf = 1.0; Pe = 10.0
    valid = [
        validar_termico_tubo(Re, Pr, Nu_rep),
        validar_masico(Re, Sc, Sh_rep, "tubo_interno")
    ]
    valores = {
        "Re": Re, "Pr": Pr, "Sc": Sc, "Pe": Pe,
        "Nu": Nu_rep, "Sh": Sh_rep, "Bi": Bi,
        "Da_ext": Da_ext, "Da_ldf": Da_ldf,
        "h": h, "k_c": k_c, "k_LDF": 1.0,
        "D_ij": 2.0e-5, "D_kn": 1.0e-5, "D_eff": 6e-6, "D_z": 2e-4
    }
    ctx = {"geometria": "tubo_interno",
           "descripcion": "Régimen intermedio/transición; upwind/TVD; incluir convección y difusión"}
    titulo = "Caso C — Régimen intermedio (transición), Pe ~ 10"
    return titulo, valores, ctx, valid

# ===========
# Ejecutable
# ===========

def escribir_archivo(nombre: str, contenido: str) -> None:
    with open(nombre, "w", encoding="utf-8") as f:
        f.write(contenido)

def generar_informes_ejemplos() -> None:
    mags = build_magnitudes()
    for make in [ejemplo_A, ejemplo_B, ejemplo_C]:
        titulo, valores, ctx, valid = make()
        informe = generar_informe_markdown(titulo, mags, valores, ctx, valid)
        if "Caso A" in titulo:
            filename = "informe_casoA.md"
        elif "Caso B" in titulo:
            filename = "informe_casoB.md"
        else:
            filename = "informe_casoC.md"
        escribir_archivo(filename, informe)
        print(f"[OK] Generado: {filename}  —  {ctx.get('descripcion','')}")

if __name__ == "__main__":
    generar_informes_ejemplos()

    # === Uso con TUS DATOS (ejemplo) ===
    # Sustituye los valores por los que ya calculaste en tu columna/adsorbedor.
    # valores = {
    #     "Re": 458.6, "Pr": 0.771, "Sc": [0.2966,0.5552], "Pe": 353.5,
    #     "Nu": 41.86, "Sh": [2.89,3.10], "Bi": 0.2425,
    #     "Da_ext": 5.9e5, "Da_ldf": [2.4e4, 1.3e4],
    #     "h": 41.86*0.018/0.02, "k_c": [0.0179,0.0313],
    #     "k_LDF": [359.3,192.3], "D_ij": 4.4e-6, "D_kn": 3.16e-3,
    #     "D_eff": [4.85e-6, 2.60e-6], "D_z": 2.92e-8
    # }
    # ctx = {"geometria": "tubo_interno", "descripcion": "Datos de columna/adsorbedor"}
    # md = generar_informe_markdown("Mi Caso — Adsorbedor", build_magnitudes(), valores, ctx)
    # escribir_archivo("informe_mi_caso.md", md)
