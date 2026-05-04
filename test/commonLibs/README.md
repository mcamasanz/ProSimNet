# Tests — CommonLibs

Propiedades de gases puros, mezclas gaseosas, sólidos estructurales y combustibles.
No integran ODEs; validan los módulos de física compartidos por todos los equipos.

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `test_commonLibs_00_getFluidProperties.ipynb` | Técnico | Verifica propiedades termofísicas de 8 especies gaseosas puras (N₂, O₂, H₂, CH₄, C₂H₆, CO₂, CO, H₂O) en modos constante y polinomial desde la base de datos de gases. |
| `test_commonLibs_01_getMixProperties.ipynb` | Técnico | Valida propiedades de mezcla (regla de Wilke para µ y λ), difusión molecular (Chapman-Enskog), difusión en poro (Knudsen + Bosanquet) y coeficientes de transferencia de calor y masa. |
| `test_commonLibs_02_getMixProperties_benchmark.ipynb` | Benchmark | Mide el coste computacional de propiedades de mezcla y coeficientes de transporte para 6 combinaciones de modos (gas_prop × transport) con N=101 nodos. |
| `test_commonLibs_03_getSolidProperties.ipynb` | Técnico | Verifica propiedades de 7 materiales sólidos estructurales (SS316L, Al₂O₃, SiC, etc.) en modos constante, polinomial y fijo, incluyendo comportamiento de clipping. |
| `test_commonLibs_04_getFuelProperties.ipynb` | Técnico | Carga propiedades de un combustible sólido (abeto) desde la base de datos: rendimientos de pirólisis, cinética de 5 reacciones, poderes caloríficos y entalpía de pirólisis. |
