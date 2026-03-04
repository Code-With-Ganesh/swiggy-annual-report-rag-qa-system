# text_chunker.py - cleans OCR text and splits into overlapping chunks

import re

def clean_ocr_text(text):
    """removes common OCR artifacts and cleans up formatting"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # non-ascii junk from OCR
    text = re.sub(r'\s[^\w\s]\s', ' ', text)    # stray symbols
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # drop very short lines (usually page numbers or noise)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 2]
    return '\n'.join(lines).strip()

def make_chunks(pages, chunk_sz=400, overlap=80):
    """splits page texts into overlapping word-level chunks"""
    chunks = []
    cid = 0
    for page in pages:
        pnum = page["page_num"]
        cleaned = clean_ocr_text(page["text"])
        words = cleaned.split()

        if len(words) <= chunk_sz:
            chunks.append({"chunk_id": cid, "text": " ".join(words), "page_num": pnum})
            cid += 1
            continue

        pos = 0
        while pos < len(words):
            chunk_words = words[pos:pos + chunk_sz]
            chunks.append({"chunk_id": cid, "text": " ".join(chunk_words), "page_num": pnum})
            cid += 1
            pos += chunk_sz - overlap

    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


if __name__ == "__main__":
    import os
    from pdf_loader import load_pdf
    pdf_path = os.path.join("data", "Swiggy Annual-Report-FY-2023-24.pdf")
    pages = load_pdf(pdf_path)
    chunks = make_chunks(pages)
    for c in chunks[:3]:
        print(f"\nChunk {c['chunk_id']} (Page {c['page_num']}) - {len(c['text'].split())} words")
        print(c["text"][:300] + "...")
