import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

def ocr_selected_pages_separately(pdf_path, output_dir, selected_pages):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    for page_num in selected_pages:
        print(f"Processing page {page_num}...")

        # Get the page (0-indexed)
        page = doc.load_page(page_num - 1)

        # Render to image
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Run OCR
        text = pytesseract.image_to_string(img)

        # Define output file path (e.g., page_51.txt)
        output_file_path = os.path.join(output_dir, f"page_{page_num}.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved OCR text of page {page_num} to {output_file_path}")

if __name__ == "__main__":
    pdf_path = '/Users/keane/Desktop/research/human_behaviour/human-behavior/data/autism_diagnostics.pdf'
    output_dir = '/Users/keane/Desktop/research/human_behaviour/human-behavior/data/ocr_pages'
    selected_pages = list(range(51, 73))  # inclusive of page 72
    ocr_selected_pages_separately(pdf_path, output_dir, selected_pages)