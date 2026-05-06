import pypdf, os
BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "A0_AncaCouce_2021_updraft_astillas_validacion", "AncaCouce_2021_Fuel_updraft_gasification.pdf")
out  = os.path.join(BASE, "_txt_A0.txt")
reader = pypdf.PdfReader(path)
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Pages: {len(reader.pages)}\n\n")
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        f.write(f"\n{'='*60}\nPage {i+1}\n{'='*60}\n{txt}")
print(f"OK: {len(reader.pages)} pages -> {out}")
