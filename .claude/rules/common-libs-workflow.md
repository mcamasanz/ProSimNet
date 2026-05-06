# Workflow para añadir especies a gasdb.txt y soliddb.txt

## Principio

`materials/fluids/gasdb.txt` y `materials/solids/soliddb.txt` son recursos compartidos
por todos los equipos. **Solo el agente de librerías comunes los modifica, siempre desde `main`.**

Los branches de equipo (`dev/reactor`, `dev/gasifier`, etc.) nunca tocan estos archivos.

---

## Quién añade una nueva especie

El **agente de librerías comunes** es el único responsable de:
1. Calcular los coeficientes polinomiales para la nueva especie.
2. Hacer el commit en `main`.

Los branches de equipo nunca modifican gasdb.txt ni soliddb.txt directamente.

---

## Cómo se calculan los coeficientes

### Cp — desde la base de datos de Fluent

Fluent proporciona polinomios de Cp en **dos rangos en base T**:

```
Cp(T) = a0 + a1·T + a2·T² + a3·T³ + a4·T⁴   [J/kg/K]
        rango 298–1000 K   →   coef_low
        rango 1000–5000 K  →   coef_high
```

El script `materials/fluids/_gen_themoLibs_cp_coeff.py`:
1. Define la función por tramos con los coeficientes de Fluent.
2. Genera datos sintéticos (298–5000 K, ~1200 puntos).
3. Ajusta un único polinomio de grado 7 en base `(T − Tref)` con `numpy.polyfit`.
4. Imprime los coeficientes `a0..a7` para gasdb.txt.

### µ y k — desde parámetros de Chapman-Enskog

Con los parámetros de Lennard-Jones `σ [Å]` y `ε/k_B [K]` de la especie:

**Viscosidad dinámica** (Chapman-Enskog):
```
µ(T) = 2.6693×10⁻⁶ · √(M·T) / (σ² · Ω^(2,2)(T*))   [Pa·s]
T*   = T / (ε/k_B)
```

Integral de colisión Ω^(2,2) — correlación de Neufeld (1972):
```
Ω = A/T*^B + C/exp(D·T*) + E/exp(F·T*)
    A=1.16145, B=0.14874, C=0.52487, D=0.7732, E=2.16178, F=2.43787
```

**Conductividad térmica** (Eucken modificada):
```
k(T) = µ(T) · (Cp(T) + 5/4 · R/M)   [W/m/K]
```

Se evalúa µ(T) y k(T) en un grid 298–5000 K y se ajusta el polinomio grado 7.

### h, u, s — desde Cp integrado + NIST/JANAF

```
h(T) = ∫[Tref→T] Cp_molar(T') dT'  +  [H°(Tref) − H°(0)]    [J/mol]
u(T) = h(T) − R·T                                               [J/mol]  (gas ideal)
s(T) = ∫[Tref→T] Cp_molar(T')/T' dT'  +  S°(Tref)            [J/mol/K]
```

Los valores de referencia `H°(Tref) − H°(0)` y `S°(Tref)` vienen de NIST WebBook / JANAF.

### Formato final en gasdb.txt

Una línea NDJSON por especie al final del archivo:

```json
{"formula":"XX","name_es":"...","molar_mass":{"value":...,"units":"kg/mol"},
 "transport":{"lj":{"sigma_A":...,"epsilon_over_k_K":...}},
 "reference":{"Tref_K":298},"limits":{"Tmax_K":5000},
 "cap_at_tmax":{...},
 "polynomials":{"basis":"deltaT","degree_max":7,"properties":{
   "mu":{"units":"Pa*s","a0_to_a7":[...]},
   "cp":{"units":"J/kg/K","a0_to_a7":[...]},
   "k": {"units":"W/m/K","a0_to_a7":[...]},
   "u": {"units":"J/mol","a0_to_a7":[...]},
   "h": {"units":"J/mol","a0_to_a7":[...]},
   "s": {"units":"J/mol/K","a0_to_a7":[...]}
 }}}
```

---

## Workflow git para añadir una especie

```bash
# 1. Cambiar a main (siempre)
git checkout main

# 2. Añadir la línea NDJSON al FINAL de gasdb.txt
#    (nunca reordenar ni tocar líneas existentes)

# 3. Commit
git add materials/fluids/gasdb.txt
git commit -m "feat(gasdb): add <formula> — LJ params + Cp from Fluent + C-E transport"

# 4. Sincronizar las ramas de equipo que necesiten la nueva especie
git checkout dev/<equipo>
git merge main
git checkout main
```

## Regla de merge para gasdb.txt

El formato NDJSON (una especie = una línea) hace que los conflictos sean triviales:
- Si dos branches añadieron especies distintas al final: **aceptar siempre las dos líneas**.
- Nunca descartar una especie existente.
- Para evitar conflictos: sincronizar con main ANTES de añadir (`git merge main`).
