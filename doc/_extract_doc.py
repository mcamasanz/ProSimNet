"""Extrae texto de los PDFs de doc/ para lectura por Claude"""
import os, pypdf

BASE = os.path.dirname(os.path.abspath(__file__))

PDFS = {
    "contrato":  "20250213. P9. CIRCE. Biochar. Contrato_vf.pdf",
    "E1_1_caract": "E1.1 - Informe de caracterización del material bioestabilizado_v1.0.pdf",
    "E2_1_tecno":  "E2.1 - Estudio de evaluación tecnológica_v1.0.pdf",
    "E3_1_modelo": "E3.1 - Desarrollo del modelo teórico de un gasificador updraft_v1.0.pdf",
    "E4_1_biochar":"E4.1 - Identificación de requisitos de calidad del biochar y casos de uso_v1.0.pdf",
}

for key, fname in PDFS.items():
    path = os.path.join(BASE, fname)
    out  = os.path.join(BASE, f"_txt_{key}.txt")
    if not os.path.exists(path):
        print(f"NOT FOUND: {fname}"); continue
    reader = pypdf.PdfReader(path)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"FILE: {fname}\nPages: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            f.write(f"\n{'='*60}\nPage {i+1}\n{'='*60}\n{txt}")
    print(f"OK {key}: {len(reader.pages)} pages -> _txt_{key}.txt")
