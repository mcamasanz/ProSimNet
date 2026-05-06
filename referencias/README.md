# Referencias — ProSimNet

Catálogo de artículos científicos organizados por equipo de simulación.
Cada subcarpeta de equipo contiene su propia estructura de artículos y case cards.

## Estructura

```
referencias/
└── gasifier/          ← Referencias para validación del gasificador
    ├── A0–A3/         Modelos 1D con validación experimental completa
    ├── B1–B5/         Cinéticas de pirólisis/gasificación de biorresiduos
    ├── C1–C3/         Propiedades térmicas del sólido (Cp, k, ρ)
    ├── D1/            Revisiones de referencia
    ├── E3/            Documento teórico interno CIRCE
    └── README.md      Catálogo detallado del gasificador
```

## Añadir un nuevo equipo

Crear subcarpeta `referencias/<equipo>/` con la misma estructura:
- `README.md` — catálogo con tabla de prioridades
- `<código>_<AutorAño>_<descripcion>/README.md` — ficha del artículo
- `<código>_<AutorAño>_<descripcion>/case_card_ProSimNet.md` — datos para simulación
- PDF del artículo cuando esté disponible

Ver `.claude/rules/validation-from-articles.md` para la metodología completa.
