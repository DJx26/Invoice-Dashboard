import os
import re
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image
import pytesseract
import pdfplumber
import PyPDF2


# -------------------------------
# Configuration (match app.py)
# -------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "invoices.db")


# -------------------------------
# Database Helpers
# -------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_name TEXT,
        invoice_date TEXT,
        amount REAL,
        invoice_number TEXT
    );
    """
    )
    conn.commit()
    return conn


# -------------------------------
# OCR Parsing (mirrors logic in app.py)
# -------------------------------
DATE_PATTERNS = [
    r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",  # dd/mm/yyyy or dd-mm-yyyy
    r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",  # yyyy-mm-dd or yyyy/mm/dd
    r"\b\d{1,2}\s*[-]\s*[A-Za-z]{3,9}\s*[-]\s*\d{4}\b",  # 12-Jun-2021
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",  # 1 Jan 2024 / 01 January 2024
    r"(?:dated|date)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # Dated: 12-Jun-2021
]

AMOUNT_PATTERNS = [
    r"(?:total\s+amount|amount\s+chargeable)\s*:?\s*(?:Rs\.?|₹)?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)\b",
    r"(?:rs\.?|₹)\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)\b",
    r"(?<!\w)(?:INR|Rs\.?|₹)?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)\b",
    r"(?<!\w)([0-9]+(?:\.[0-9]{2})?)\b"
]


def parse_date(text: str) -> str:
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            raw = m.group(0)
            # Handle patterns with capture groups
            if m.groups():
                raw = m.group(1)
            
            # Try various date formats
            date_formats = [
                "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%d %b %Y", "%d %B %Y", "%-d %b %Y", "%-d %B %Y",
                "%d-%b-%Y", "%d-%B-%Y",  # For formats like 12-Jun-2021
            ]
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(raw, fmt)
                    return dt.date().isoformat()
                except Exception:
                    continue
            return raw
    return ""


def parse_amount(text: str) -> float:
    candidates = []
    for pat in AMOUNT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            val = m.group(1)
            if not val:
                continue
            val = val.replace(",", "")
            try:
                candidates.append(float(val))
            except Exception:
                pass
    return max(candidates) if candidates else 0.0

def parse_invoice_number(text: str) -> str:
    # Look for invoice number patterns specific to Indian invoices
    invoice_patterns = [
        r"(?:invoice\s+no\.?|inv\.?\s*no\.?)\s*:?\s*([A-Z0-9\-_/]+)",
        r"(?:invoice\s+number|inv\.?\s*number)\s*:?\s*([A-Z0-9\-_/]+)",
        r"(?:bill\s+no\.?|receipt\s+no\.?)\s*:?\s*([A-Z0-9\-_/]+)",
        r"([A-Z]{2,4}[/-]\d{4}-\d{2}/\d{2,3})",  # Format like TI/2021-22/068
        r"([A-Z]{2,4}\d{4,8})",  # Format like INV2024001
        r"(\d{4,8})",  # Just numbers as fallback
    ]
    
    for pattern in invoice_patterns:
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            invoice_num = match.group(1).strip()
            if len(invoice_num) >= 3:  # Minimum length for invoice number
                return invoice_num
    return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file using multiple methods"""
    try:
        # Method 1: Try pdfplumber first (better for text extraction)
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        
        # Method 2: Fallback to PyPDF2
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def guess_seller_name(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    # Look for structured invoice patterns first
    for i, line in enumerate(lines):
        # Check for "Tax Invoice" header and look for seller info below
        if re.search(r"tax\s+invoice", line, flags=re.IGNORECASE):
            # Look for seller information in next few lines
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j].strip()
                # Skip empty lines and common headers
                if not next_line or re.search(r"(invoice|gstin|address|state|code)", next_line, flags=re.IGNORECASE):
                    continue
                # Look for company name patterns
                if re.search(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", next_line) and len(next_line) < 100:
                    return next_line
    
    # Look for specific seller patterns in Indian invoices
    seller_patterns = [
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:International|Limited|Ltd|Pvt|Company|Co))",
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # General company name
    ]
    
    for line in lines[:20]:
        for pattern in seller_patterns:
            match = re.search(pattern, line)
            if match:
                seller_name = match.group(1).strip()
                # Skip if it's clearly not a seller name
                if not re.search(r"(invoice|tax|gst|bill|receipt|date|amount|total|due|paid|buyer|consignee)", seller_name, flags=re.IGNORECASE):
                    return seller_name
    
    # Skip common invoice headers and metadata
    skip_patterns = [
        r"(invoice|tax|gst|bill|receipt|date|amount|total|due|paid|seller|buyer|consignee)",
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",  # Date patterns
        r"^\d+\.?\d*$",  # Just numbers
        r"^[A-Z\s]+$",  # All caps (likely headers)
        r"^\s*$",  # Empty lines
        r"^[A-Za-z]+\s*:$",  # Labels ending with colon
    ]
    
    for ln in lines[:15]:
        # Skip if matches any skip pattern
        if any(re.search(pattern, ln, flags=re.IGNORECASE) for pattern in skip_patterns):
            continue
            
        # Must have at least 3 alphabetic characters
        if len(re.sub(r"[^A-Za-z]", "", ln)) < 3:
            continue
            
        # Should not be too long (likely not a company name)
        if len(ln) > 100:
            continue
            
        # Should contain some letters and not be all numbers/symbols
        if re.search(r"[A-Za-z]", ln) and not re.match(r"^[\d\s\-\.\/]+$", ln):
            return ln.strip()
    
    # Fallback: return first non-empty line that looks like a name
    for ln in lines:
        if len(ln.strip()) > 2 and re.search(r"[A-Za-z]", ln):
            return ln.strip()
    
    return "Unknown Seller"


# -------------------------------
# Import Routine
# -------------------------------
def import_files(root_dir: Path, limit: int | None = None) -> int:
    conn = init_db()
    cur = conn.cursor()

    # Collect all supported file paths
    file_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.pdf"]:
        file_paths.extend(root_dir.rglob(ext))

    # Stable order for reproducibility
    file_paths.sort()

    inserted = 0
    for idx, file_path in enumerate(file_paths, start=1):
        if limit is not None and inserted >= limit:
            break
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png']:
                # Process image with OCR
                image = Image.open(file_path).convert("RGB")
                ocr_text = pytesseract.image_to_string(image)
                file_type = "Image"
            elif file_extension == '.pdf':
                # Process PDF with text extraction
                ocr_text = extract_text_from_pdf(file_path)
                file_type = "PDF"
            else:
                continue

            seller_name = guess_seller_name(ocr_text)
            invoice_date = parse_date(ocr_text)
            amount = parse_amount(ocr_text)
            invoice_number = parse_invoice_number(ocr_text)

            cur.execute(
                "INSERT INTO invoices (seller_name, invoice_date, amount, invoice_number) VALUES (?,?,?,?)",
                (seller_name, invoice_date, float(amount), invoice_number),
            )
            inserted += 1
            if inserted % 25 == 0:
                conn.commit()
                print(f"Committed {inserted} records so far...")
        except Exception as e:
            print(f"[WARN] Skipping {file_path}: {e}")
            continue

    conn.commit()
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Import invoice images into SQLite DB using OCR")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).parent / "DATASET"),
        help="Root directory containing invoice images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    print(f"Using database: {DB_PATH}")
    print(f"Scanning for invoice files (images & PDFs) under: {root_dir}")

    count = import_files(root_dir, limit=args.limit)
    print(f"Imported {count} invoices.")


if __name__ == "__main__":
    main()


