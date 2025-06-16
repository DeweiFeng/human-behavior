import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def ocr_selected_pages(pdf_path, output_txt_path, selected_pages):
    """
    selected_pages: list of 1-based page numbers to OCR, e.g., [1, 3, 5]
    """
    # Convert only selected pages (note: 0-based index internally)
    page_indices = [p - 1 for p in selected_pages]
    images = convert_from_path(pdf_path, dpi=300, first_page=min(selected_pages), last_page=max(selected_pages))
    
    full_text = ""

    for i, page_num in enumerate(range(min(selected_pages), max(selected_pages) + 1)):
        if page_num in selected_pages:
            print(f"Processing page {page_num}...")
            text = pytesseract.image_to_string(images[i])
            full_text += f"\n\n=== Page {page_num} ===\n\n{text}"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Text extracted for pages {selected_pages} and saved to {output_txt_path}")

if __name__ == "__main__":
    # Example usage
    pdf_path = '/home/keaneong/human-behavior/data/autism_diagnostics.pdf'
    output_txt_path = '/home/keaneong/human-behavior/data/trial_text_output.txt'
    selected_pages = [1, 3, 5]  # <- 1-based page numbers
    ocr_selected_pages(pdf_path, output_txt_path, selected_pages)