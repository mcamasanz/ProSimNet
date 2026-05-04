# CLAUDE.local.md — Preferencias personales del desarrollador

> Este archivo es personal y NO debe subirse al repositorio.
> Añadir a .gitignore: `CLAUDE.local.md`
>
> Contiene preferencias del desarrollador que complementan CLAUDE.md y
> los archivos de .claude/ sin sobreescribirlos.

---

## Preferencias de respuesta

- Respuestas concisas y directas. Sin introducción, sin resumen final redundante.
- Español en toda la comunicación con el usuario.
- Código en inglés técnico, comentarios en español cuando aporten valor.
- Si hay duda sobre el alcance de un cambio, preguntar antes de implementar.

---

## Contexto del desarrollador

- Ingeniero de proceso con experiencia en simulación numérica industrial.
- Familiarizado con transferencia de calor, masa, termodinámica, fluidos, cinética.
- No necesita explicaciones de conceptos básicos de ingeniería química o Python.
- Prefiere ver el análisis físico/matemático antes del código cuando el cambio es profundo.

---

## Flujo de trabajo preferido

1. Para cambios de física o arquitectura: análisis primero, código después.
2. Para bugs simples o correcciones menores: código directamente.
3. Para nuevos equipos: usar `/new-equipment` como punto de partida.
4. Para auditar RHS: usar `/check-rhs`.
5. Commits atómicos con mensajes descriptivos del "por qué".

---

## Entorno de desarrollo

- Editor: VS Code con extensión Claude Code
- Python: entorno conda/venv con numpy, scipy, matplotlib, jupyter
- Sistema operativo: Windows 11
- Ruta del proyecto: `c:\Users\MiguelCamaraSanz\OneDrive - Fundacion CIRCE\GITHUB\ProSimNet`
- Nombre del proyecto: **ProSimNet** (Process Simulation Network)

---

## Notas personales activas

<!-- Añadir aquí notas de sesión, decisiones pendientes, recordatorios -->

- Validación contra modelo1dcinetico.pdf — objetivo final del gasificador
- WGS y tar cracking pendientes de implementar (reacciones homogéneas)
- Convección sólida en balance de energía (div(vs·Cp_s·Ts)) pendiente para conveyor
