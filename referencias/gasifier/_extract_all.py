"""Extrae texto de todos los PDFs de referencias del gasificador y los guarda como .txt"""
import os, pypdf

BASE = os.path.dirname(os.path.abspath(__file__))

PDFS = {
    "A0": "A0_AncaCouce_2021_updraft_astillas_validacion/AncaCouce_2021_Fuel_updraft_gasification.pdf",
    "A1": "A1_DiBlasi_2004_gasificacion_updraft_madera/DiBlasi_2004_AIChE.pdf",
    "A2": "A2_Chen_2020_residuo_jardin_pirolisis_gasificacion/Chen_2020_Energy.pdf",
    "A3": "A3_CoGasificacion_2022_updraft_lecho_fijo/CoGasificacion_2022_Fuel.pdf",
    "B1": "B1_MSW_2023_pirolisis_cinetica_multistep/MSW_2023_WasteManagement.pdf",
    "B2": "B2_LodoDepuradora_2021_pirolisis_cinetica/LodoDepuradora_2021_JEnvManagement.pdf",
    "B3": "B3_LodoDepuradora_2019_bioenergy/LodoDepuradora_2019_RSER.pdf",
    "E3": "E3_ModeloTeorico_Updraft_v1/ModeloTeorico_Updraft_v1.0.pdf",
}

for key, rel in PDFS.items():
    path = os.path.join(BASE, rel)
    out  = os.path.join(BASE, f"_txt_{key}.txt")
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}"); continue
    reader = pypdf.PdfReader(path)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Pages: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            f.write(f"\n{'='*60}\nPage {i+1}\n{'='*60}\n{txt}")
    print(f"OK {key}: {len(reader.pages)} pages -> {os.path.basename(out)}")
