# pdf_loader.py - extracts text from all pages of a PDF
# uses PyMuPDF for normal text pages and Tesseract for scanned/image pages

import fitz
import pytesseract
from PIL import Image
import io, os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_page_text(page):
    """try direct extraction first, fall back to OCR if the page is scanned"""
    text = page.get_text().strip()
    if len(text) > 50:
        return text
    # probably a scanned page, render and OCR it
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img).strip()

def load_pdf(filepath):
    """reads entire PDF, returns list of {page_num, text} dicts"""
    doc = fitz.open(filepath)
    total = len(doc)
    print(f"Processing {total} pages from: {filepath}")

    pages = []
    txt_count, ocr_count = 0, 0

    for i in range(total):
        pg = doc[i]
        raw = pg.get_text().strip()
        if len(raw) > 50:
            text = raw
            txt_count += 1
        else:
            pix = pg.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()
            ocr_count += 1

        if text and len(text) > 20:
            pages.append({"page_num": i + 1, "text": text})

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{total} pages...")

    doc.close()
    print(f"Done! {len(pages)} pages extracted ({txt_count} text + {ocr_count} OCR)")
    return pages


if __name__ == "__main__":
    pdf_path = os.path.join("data", "Swiggy Annual-Report-FY-2023-24.pdf")
    result = load_pdf(pdf_path)
    for p in result[:3]:
        print(f"\n--- Page {p['page_num']} ---")
        print(f"Words: {len(p['text'].split())}")
        print(p["text"][:400] + "...")
