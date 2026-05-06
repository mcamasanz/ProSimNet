# Modos de operación del gasificador — contextualización conceptual

## 1. Qué significa "modo de operación"

El modo de operación no es una etiqueta arbitraria. Es una descripción compacta de cómo interactúan
el gas, el sólido y el dominio espacial. Queda definido por la combinación de:

1. Dimensionalidad del modelo: 0D (N=1) o 1D (N>1)
2. Comportamiento del gas: ausente, producido internamente, inyectado con dirección
3. Comportamiento del sólido: fijo, con transporte por gravedad, con transporte mecánico
4. Condiciones de entrada y salida de cada fase
5. Dirección relativa de movimiento entre fases (co-corriente, contra-corriente)

---

## 2. Dimensionalidad: 0D vs 1D

### 0D — N=1

El reactor se representa como un único volumen perfectamente mezclado.
No existe una dirección espacial interna ni caras entre volúmenes de control.

**Consecuencias:**
- No hay velocidades de cara internas (no hay caras)
- Las únicas condiciones de contorno son la entrada y salida del reactor como conjunto
- Los gradientes axiales no pueden representarse: temperatura, concentración y densidad del sólido
  son uniformes en todo el volumen en cada instante
- Representa físicamente: retortas, reactores CSTR, hornos perfectamente mezclados

### 1D — N>1

El reactor se representa como una sucesión de N volúmenes de control a lo largo de una dirección axial.
Entre volúmenes existen N+1 caras.

**Consecuencias:**
- Las velocidades de fase (gas, sólido) viven en las caras
- Si una fase no se mueve, la velocidad en caras es cero pero existe como variable de transporte
- Pueden representarse gradientes axiales de temperatura, composición y densidad
- Representa físicamente: gasificadores de lecho fijo con combustible cargado axialmente

---

## 3. Velocidades en caras: 0D vs 1D

| Situación | 0D (N=1) | 1D (N>1) |
|-----------|----------|----------|
| Gas en movimiento | No aplica — no hay caras | v_face[i] ≠ 0 en caras i=0..N |
| Gas sin movimiento | No aplica | v_face[i] = 0 en todas las caras |
| Sólido en movimiento | No aplica | vs_face[i] ≠ 0 en caras de sólido |
| Sólido fijo | No aplica | vs_face[i] = 0 en todas las caras |

La velocidad en caras puede existir como variable con valor cero en 1D.
En 0D no debe usarse para describir el proceso (no hay geometría axial).

---

## 4. Comportamiento del gas (eje 1)

| Tipo | Descripción | bc_config |
|------|-------------|-----------|
| **Batch** | Sin gas externo. El gas aparece únicamente por reacciones internas (pirólisis, gasificación). | `v_gas_in=None, outlet="open"` |
| **Semibatch** | Sin gas externo, pero con alivio de presión controlado. El gas producido puede ventear si P > P_out. | `v_gas_in=None, outlet="vent"` |
| **CSTR** | Gas inyectado con flujo constante. Perfectamente mezclado con el gas del reactor. | `v_gas_in=float, outlet="open"` |
| **Updraft** | Gas entra por la parte inferior y sale por la parte superior (en contra del sólido). | `v_gas_in=float, direction="updraft"` |
| **Downdraft** | Gas entra por la parte superior y sale por la parte inferior (en el mismo sentido que el sólido). | `v_gas_in=float, direction="downdraft"` |

---

## 5. Comportamiento del sólido (eje 2)

| Tipo | Descripción | bc_config |
|------|-------------|-----------|
| **Batch** | Sólido cargado inicialmente y fijo. No entra ni sale sólido durante la simulación. | `v_solid=0.0` |
| **Updraft / Downdraft** | Sólido que se desplaza por gravedad en dirección opuesta o igual al gas. Requiere condición de entrada del sólido. | `v_solid=float, direction="updraft"/"downdraft"` |
| **Conveyor** | Sólido transportado mecánicamente (tornillo sin fin). El tiempo de residencia del sólido es impuesto externamente. | `v_solid=float, inlet_mode="computed"` |

---

## 6. Combinaciones modo gas / modo sólido

| Combinación | Físico | 0D | 1D |
|-------------|--------|----|----|
| Batch gas / Batch sólido | Retorta, horno de pirólisis cerrado | ✓ | ✓ |
| Semibatch gas / Batch sólido | Retorta con alivio de presión | ✓ | ✓ |
| CSTR gas / Batch sólido | Lecho fijo con inyección de gas | ✓ | ✓ |
| Updraft gas / Updraft sólido | Gasificador de tiro natural en contra-corriente | ✗ | ✓ |
| Downdraft gas / Downdraft sólido | Gasificador de lecho móvil en co-corriente | ✗ | ✓ |
| CSTR/Downdraft gas / Conveyor sólido | Gasificador con tornillo transportador | ✗ | ✓ |

En 0D, los modos con sólido en movimiento no tienen sentido (no existe dirección espacial).

---

## 7. Rol de las condiciones de contorno

Las BC definen el modo físico. La misma geometría puede representar equipos diferentes
según las condiciones de contorno:

- `v_gas_in=None` → la biomasa evoluciona sola (pirolizador)
- `v_gas_in=float, air` → el char se oxida (gasificador)
- `v_gas_in=float, steam` → gasificación con vapor
- `v_solid > 0, updraft` → lecho móvil en contra-corriente (gasificador tipo Lurgi)
- `v_solid > 0, conveyor` → reactor de tornillo (residence time controlado)

---

## 8. Serie de tests — progresión didáctica

La serie va de lo más simple a lo más complejo, con un test por concepto:

### Bloque 1 — Reactor concentrado 0D
| Test | Modo | Concepto clave |
|------|------|----------------|
| `test_gasifier_01_0D_batch.ipynb` | 0D Batch | Base de todo: pirólisis en reactor cerrado |
| `test_gasifier_02_0D_semibatch.ipynb` | 0D Semibatch | Efecto del alivio de presión en la pirólisis |
| `test_gasifier_03_0D_cstr.ipynb` | 0D CSTR | Inyección de agente gasificante en reactor mezclado |

### Bloque 2 — Lecho fijo 1D
| Test | Modo | Concepto clave |
|------|------|----------------|
| `test_gasifier_10_1D_batch.ipynb` | 1D Batch | Gradientes axiales sin flujo de gas |
| `test_gasifier_11_1D_cstr.ipynb` | 1D CSTR | Gas a través de lecho fijo (0D→1D comparación) |

### Bloque 3 — Gasificador con flujo
| Test | Modo | Concepto clave |
|------|------|----------------|
| `test_gasifier_12_1D_updraft.ipynb` | 1D Updraft | Contra-corriente gas↑ sólido↓ |
| `test_gasifier_13_1D_downdraft.ipynb` | 1D Downdraft | Co-corriente gas↓ sólido↓ |
| `test_gasifier_14_1D_conveyor.ipynb` | 1D Conveyor | Tiempo de residencia del sólido impuesto |

### Bloque 4 — Térmica avanzada
| Test | Concepto clave |
|------|----------------|
| `test_gasifier_20_wall_models.ipynb` | Comparación de 4 modos de BC térmico |
| `test_gasifier_21_shell_tube.ipynb` | Pared dinámica con modelo ODE |

### Bloque 5 — Validación y análisis
| Test | Concepto clave |
|------|----------------|
| `test_gasifier_30_balances.ipynb` | Cierre de balances masa y energía en todos los modos |
| `test_gasifier_31_convergence_0D_1D.ipynb` | Convergencia espacial N=1 → N=50 |

### Bloque 6 — Optimización (futuro)
| Test | Concepto clave |
|------|----------------|
| `test_gasifier_40_parametric.ipynb` | Estudio paramétrico de variables de diseño |
| `test_gasifier_50_surrogate.ipynb` | Modelo sustituto (ROM) |
| `test_gasifier_60_nn_optimization.ipynb` | Optimización con redes neuronales |

---

## 9. Ambigüedades a evitar

1. **"Batch" no siempre significa "cerrado"**: en este modelo, batch/gas significa sin flujo forzado
   de gas externo. El gas producido puede evacuar para mantener la presión.

2. **"Updraft" y "downdraft" describen la dirección del GAS respecto a la gravedad**,
   no necesariamente la dirección del sólido. Confirmar siempre con el diagrama de flujo.

3. **CSTR ≠ 0D si N>1**: CSTR se refiere al comportamiento del gas (entrada continua),
   no a la dimensionalidad del modelo. Un CSTR puede ser 1D si N>1.

4. **Velocidad de cara no implica movimiento neto**: en 1D una velocidad de cara puede
   representar transporte convectivo con valor cero (sólido fijo). No confundir con ausencia
   de la variable.

5. **Los modos con sólido en movimiento solo tienen sentido en 1D**: updraft, downdraft y
   conveyor requieren una dirección axial. En 0D (N=1) no existen caras y no puede haber
   transporte de sólido.
