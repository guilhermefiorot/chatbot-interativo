import pdfplumber
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert('L')
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    # Optional: sharpen
    img = img.filter(ImageFilter.SHARPEN)
    return img

def extract_cnh_fields_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        img = preprocess_image(img)
        text += pytesseract.image_to_string(img, lang="por") + "\n"

    # Extract MRZ-based registration number
    registro = None
    mrz_match = re.search(r'I<BRA([A-Z0-9<]+)', text)
    if mrz_match:
        mrz_line = mrz_match.group(0)
        mrz_line = correct_mrz_ocr(mrz_line)
        value = mrz_line[5:17]
        registro = value.replace('<', '')
        
    print({
        "registro": registro,
        "raw_text": text
    })
    return {
        "registro": registro,
        "raw_text": text
    }

def correct_mrz_ocr(text):
    # Only correct in the MRZ zone, not the whole OCR text!
    corrections = str.maketrans({
        'D': '0',
        'O': '0',
        'Q': '0',
        'I': '1',  # Only if you see this error, otherwise comment out
        'S': '5',
        # 'B': '8',  # Uncomment if you see this error
    })
    return text.translate(corrections)

def extract_cnh_mrz_fields(text):
    # Find the MRZ line (starts with I<BRA)
    mrz_match = re.search(r'I<BRA([A-Z0-9<]+)', text)
    if mrz_match:
        mrz_line = mrz_match.group(0)
        mrz_line = correct_mrz_ocr(mrz_line)
        value = mrz_line[5:17]
        value = value.replace('<', '')
        return value
    return None