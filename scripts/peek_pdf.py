import sys
import fitz  # PyMuPDF's import name

doc = fitz.open(sys.argv[1])
print("pages:", len(doc))
for page_number in (0, 3):  # fitz counts pages from 0
    text = doc[page_number].get_text()
    print(f"--- page index {page_number} (human page {page_number + 1}) ---")
    print(text[:800])
    