import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def ocr_selected_pages_no_poppler(pdf_path, output_txt_path, selected_pages):
    """
    selected_pages: list of 1-based page numbers to OCR, e.g., [1, 3, 5]
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in selected_pages:
        print(f"Processing page {page_num}...")

        # Get the page (0-indexed in PyMuPDF)
        page = doc.load_page(page_num - 1)

        # Render page to an image
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Run OCR
        text = pytesseract.image_to_string(img)
        full_text += f"\n\n=== Page {page_num} ===\n\n{text}"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Text extracted for pages {selected_pages} and saved to {output_txt_path}")

if __name__ == "__main__":
    pdf_path = '/Users/keane/Desktop/research/human_behaviour/autism_diagnostics.pdf'
    output_txt_path = '/Users/keane/Desktop/research/human_behaviour/trial_text_output.txt'
    selected_pages = [1]
    ocr_selected_pages_no_poppler(pdf_path, output_txt_path, selected_pages)