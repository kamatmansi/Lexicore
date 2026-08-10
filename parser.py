import fitz  # PyMuPDF
import re
import os


def extract_text_from_pdf(pdf_path):
    """
    Takes a PDF file path
    Returns clean extracted text
    """

    # ── Step 1: Check file exists ───────────────────────────
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return None

    print(f"Opening: {pdf_path}")

    # ── Step 2: Open PDF and extract text ───────────────────
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")

    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        # Strip bare page numbers at the top or bottom of each page (P21).
        # Only safe here, where page boundaries are still known.
        page_text = re.sub(r'^\s*\d{1,3}\s*$', '', page_text,
                           flags=re.MULTILINE, count=1)
        page_text = re.sub(r'\n\s*\d{1,3}\s*\n?$', '\n', page_text)
        full_text += page_text + "\n"

    doc.close()

    # ── Step 3: Clean the extracted text ────────────────────
    clean = full_text

    # Fix broken words across lines (e.g. "Agree-\nment")
    clean = re.sub(r'-\n', '', clean)

    # Remove page numbers like "Page 1 of 5"
    clean = re.sub(r'page \d+ of \d+', '', clean, flags=re.IGNORECASE)

    # Remove multiple spaces
    clean = re.sub(r'[ \t]+', ' ', clean)

    # Remove excessive blank lines
    clean = re.sub(r'\n{3,}', '\n\n', clean)

    # Strip leading/trailing whitespace
    clean = clean.strip()

    return clean


def get_pdf_info(pdf_path):
    """
    Returns basic info about the PDF
    """
    doc = fitz.open(pdf_path)
    info = {
        "filename": os.path.basename(pdf_path),
        "total_pages": len(doc),
        "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2)
    }
    doc.close()
    return info


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        "data/sample_contract.pdf",
        "data/COE-Sample.pdf",
        "data/sample-service-agreement.pdf"
    ]

    for pdf_path in test_files:
        print("=" * 60)

        # Get PDF info
        info = get_pdf_info(pdf_path)
        print(f"File     : {info['filename']}")
        print(f"Pages    : {info['total_pages']}")
        print(f"Size     : {info['file_size_kb']} KB")
        print()

        # Extract text
        text = extract_text_from_pdf(pdf_path)

        if text:
            print(f"Characters extracted : {len(text)}")
            print(f"Words extracted      : {len(text.split())}")
            print()
            print("--- First 500 characters ---")
            print(text[:500])
            print()
            print("--- Last 200 characters ---")
            print(text[-200:])

        print()