
import os
import re
import sqlite3
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
import plotly.express as px
import pdfplumber
import PyPDF2
import joblib
import requests

# -------------------------------
# Configuration
# -------------------------------
DEFAULT_COMPANY_NAME = "Jenendra Press"
DB_PATH = os.path.join(os.path.dirname(__file__), "invoices.db")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "seller_model.joblib")
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback", "training_samples.jsonl")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

st.set_page_config(page_title="Invoice Management Dashboard", layout="wide")

# -------------------------------
# Database Helpers
# -------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_name TEXT,
        invoice_date TEXT,
        amount REAL,
        invoice_number TEXT
    );
    """)
    conn.commit()
    return conn

conn = init_db()

# -------------------------------
# UI Header
# -------------------------------
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists(LOGO_PATH):
        try:
            st.image(LOGO_PATH, use_container_width=True, caption="")
        except Exception as e:
            st.markdown("### 🏭 **Jenendra Press**")
    else:
        st.markdown("### 🏭 **Jenendra Press**")
with col_title:
    st.title("📊 Invoice Management Dashboard")

# Company name (editable in sidebar, stored in session)
if "company_name" not in st.session_state:
    st.session_state.company_name = DEFAULT_COMPANY_NAME

with st.sidebar:
    st.header("🏭 Company")
    st.session_state.company_name = st.text_input(
        "Company Name", value=st.session_state.company_name
    )
    st.caption("Shown on the top of reports and exports.")
    st.divider()
    st.header("🤖 Model")
    if st.button("Train/Resume Seller Model"):
        with st.spinner("Training model from DATASET... this can take a while"):
            try:
                # Lazy import to avoid overhead at app start
                import subprocess, sys
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "train_model.py"),
                    "--root", os.path.join(os.path.dirname(__file__), "dataset for training", "DATASET"),
                    "--out", MODEL_PATH,
                    "--epochs", "3",
                    "--resume",
                ]
                subprocess.check_call(cmd)
                # Reload model into session
                st.session_state["seller_model_obj"] = joblib.load(MODEL_PATH)
                st.success("Training complete. Model loaded.")
            except Exception as e:
                st.error(f"Training failed: {e}")
    st.checkbox("Use local LLM (Ollama) for extraction", key="use_llm", help="Requires Ollama running locally.")
    st.text_input("Ollama model", value=st.session_state.get("ollama_model", "llama3.2:3b"), key="ollama_model")

st.markdown(f"**Company:** {st.session_state.company_name}")

# -------------------------------
# OCR & Parsing
# -------------------------------
DATE_PATTERNS = [
    r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",      # dd/mm/yyyy or dd-mm-yyyy
    r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",      # yyyy-mm-dd or yyyy/mm/dd
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

def parse_date(text: str):
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
                    # Some formats with %-d are POSIX only, so try/catch
                    dt = datetime.strptime(raw, fmt)
                    return dt.date().isoformat()
                except Exception:
                    continue
            # Fallback: return raw if parsing failed
            return raw
    return ""

def parse_amount(text: str):
    # Look for specific total amount patterns first
    total_patterns = [
        r"total\s+amount\s*:?\s*(?:rs\.?|₹)?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)",
        r"rs\.?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)\s*only",
        r"amount\s+chargeable\s*:?\s*(?:rs\.?|₹)?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)",
        r"grand\s+total\s*:?\s*(?:rs\.?|₹)?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)",
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            val = match.group(1).replace(",", "")
            try:
                amount = float(val)
                # Very strict range for realistic invoice amounts
                if 100 <= amount <= 1000000:  # Between 100 and 1 million
                    return amount
            except Exception:
                pass
    
    # Look for amounts in specific contexts only
    context_patterns = [
        r"(?:total|amount|rs\.?|₹)\s*:?\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})?)",
    ]
    
    candidates = []
    for pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            val = match.group(1).replace(",", "")
            try:
                amount = float(val)
                # Very strict filtering
                if 100 <= amount <= 1000000:  # Between 100 and 1 million
                    # Skip phone numbers, e-way bills, etc.
                    if len(str(int(amount))) < 10:  # Should be less than 10 digits
                        candidates.append(amount)
            except Exception:
                pass
    
    return max(candidates) if candidates else 0.0

def parse_invoice_number(text: str):
    # Look for TI/2021-22/068 pattern specifically first
    ti_pattern = r"TI/\d{4}-\d{2}/\d{2,3}"
    match = re.search(ti_pattern, text)
    if match:
        return match.group(0)
    
    # Look for invoice number patterns specific to Indian invoices
    invoice_patterns = [
        r"(?:invoice\s+no\.?|inv\.?\s*no\.?)\s*:?\s*([A-Z0-9\-_/]+)",
        r"(?:invoice\s+number|inv\.?\s*number)\s*:?\s*([A-Z0-9\-_/]+)",
        r"(?:bill\s+no\.?|receipt\s+no\.?)\s*:?\s*([A-Z0-9\-_/]+)",
        r"([A-Z]{2,4}[/-]\d{4}-\d{2}/\d{2,3})",  # Format like TI/2021-22/068
        r"([A-Z]{2,4}\d{4,8})",  # Format like INV2024001
    ]
    
    for pattern in invoice_patterns:
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            invoice_num = match.group(1).strip()
            # Skip common false positives
            if invoice_num.lower() in ['way', 'bill', 'no', 'number', 'date', 'amount', 'eway', 'e-way', 'dated']:
                continue
            # Skip if it contains "way" anywhere
            if 'way' in invoice_num.lower():
                continue
            # Skip if it's just a single word that's not an invoice number
            if len(invoice_num) < 3 or len(invoice_num) > 20:
                continue
            # Skip if it's just numbers (likely not an invoice number)
            if re.match(r"^\d+$", invoice_num):
                continue
            return invoice_num
    
    return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using multiple methods"""
    try:
        # Method 1: Try pdfplumber first (better for text extraction)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        
        # Method 2: Fallback to PyPDF2
        pdf_file.seek(0)  # Reset file pointer
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def guess_seller_name(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    # Look for specific company names first
    specific_companies = [
        "Tradecon International",
        "Jenendra Press",
        "Shree Jenendra",
        "Jenendra"
    ]
    
    for line in lines:
        for company in specific_companies:
            if company.lower() in line.lower():
                return company
    
    # Look for company name after "Tax Invoice" header
    for i, line in enumerate(lines):
        if re.search(r"tax\s+invoice", line, flags=re.IGNORECASE):
            # Look for company name in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                next_line = lines[j].strip()
                # Skip empty lines and common headers
                if not next_line or re.search(r"(invoice|gstin|address|state|code|terms|payment|delivery|note|mode)", next_line, flags=re.IGNORECASE):
                    continue
                # Look for proper company name pattern
                if re.search(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", next_line) and len(next_line) < 50:
                    return next_line
    
    # Look for company name patterns
    for line in lines[:10]:
        # Skip lines that are clearly not company names
        if re.search(r"(invoice|tax|gst|bill|receipt|date|amount|total|due|paid|buyer|consignee|terms|payment|way|delivery|note|mode|contact|no|ack|state|name)", line, flags=re.IGNORECASE):
            continue
        # Skip lines that are all caps (headers)
        if line.isupper():
            continue
        # Skip lines that are just numbers
        if re.match(r"^[\d\s\-\.\/]+$", line):
            continue
        # Skip lines that are too long
        if len(line) > 50:
            continue
        # Look for proper company name format
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", line):
            return line
    
    return "Unknown Seller"

def predict_seller_with_model(text: str):
    try:
        if not os.path.exists(MODEL_PATH):
            return None
        model_obj = st.session_state.get("seller_model_obj")
        if model_obj is None:
            model_obj = joblib.load(MODEL_PATH)
            st.session_state["seller_model_obj"] = model_obj
        tfidf = model_obj.get("tfidf")
        clf = model_obj.get("clf")
        if tfidf is None or clf is None:
            return None
        X = tfidf.transform([text])
        pred = clf.predict(X)[0]
        return pred
    except Exception:
        return None

def predict_seller_with_model_confidence(text: str):
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None
        model_obj = st.session_state.get("seller_model_obj")
        if model_obj is None:
            model_obj = joblib.load(MODEL_PATH)
            st.session_state["seller_model_obj"] = model_obj
        tfidf = model_obj.get("tfidf")
        clf = model_obj.get("clf")
        if tfidf is None or clf is None:
            return None, None
        X = tfidf.transform([text])
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            pred_idx = proba.argmax()
            pred = clf.classes_[pred_idx]
            conf = float(proba[pred_idx])
            return pred, conf
        pred = clf.predict(X)[0]
        return pred, None
    except Exception:
        return None, None

def llm_extract_fields(text: str, model: str = "llama3.2:3b"):
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        prompt = (
            "Extract the following fields from the invoice text and return a compact JSON object with keys: "
            "seller_name, invoice_date, amount, invoice_number. "
            "Use ISO date (YYYY-MM-DD) if possible and numeric amount without commas.\n\n" + text
        )
        payload = {"model": model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        out = resp.json().get("response", "{}")
        # Try to locate JSON in the response
        import json, re
        match = re.search(r"\{[\s\S]*\}", out)
        if match:
            obj = json.loads(match.group(0))
        else:
            obj = json.loads(out)
        return {
            "seller_name": str(obj.get("seller_name", "")).strip(),
            "invoice_date": str(obj.get("invoice_date", "")).strip(),
            "amount": float(str(obj.get("amount", "0")).replace(",", "")) if obj.get("amount") not in (None, "") else 0.0,
            "invoice_number": str(obj.get("invoice_number", "")).strip(),
        }
    except Exception:
        return None

# -------------------------------
# Upload Section
# -------------------------------
st.header("📤 Upload New Invoice")
st.caption("Upload a JPG/PNG invoice image. The app will OCR and store Seller, Date, Amount, and Invoice Number.")
st.info("💡 **Note:** Amount extracted is typically the total invoice amount (including tax). You can edit this before saving.")

uploaded_file = st.file_uploader("Upload Invoice (jpg, png, jpeg, pdf)", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file is not None:
    try:
        # Check file type and process accordingly
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Process image with OCR
            image = Image.open(uploaded_file).convert("RGB")
            ocr_text = pytesseract.image_to_string(image)
            file_type = "Image (OCR)"
        elif file_extension == 'pdf':
            # Process PDF with text extraction
            ocr_text = extract_text_from_pdf(uploaded_file)
            file_type = "PDF (Text Extraction)"
        else:
            st.error("Unsupported file type. Please upload JPG, PNG, JPEG, or PDF files.")
            st.stop()

        with st.expander(f"📝 Extracted Text from {file_type} (click to expand)"):
            st.text_area("Extracted Text", ocr_text, height=200)

        # Optional: local LLM extraction for better fields
        llm_fields = None
        if st.session_state.get("use_llm"):
            llm_fields = llm_extract_fields(ocr_text, model=st.session_state.get("ollama_model", "llama3.2:3b"))

        # Suggestions (ML + rules)
        ml_pred, ml_conf = predict_seller_with_model_confidence(ocr_text)
        rule_pred = guess_seller_name(ocr_text)
        suggestion = (llm_fields or {}).get("seller_name") or ml_pred or rule_pred
        invoice_date = parse_date(ocr_text)
        amount = parse_amount(ocr_text)
        invoice_number = parse_invoice_number(ocr_text)

        with st.form("confirm_save"):
            st.subheader("🔎 Parsed Fields (you can edit before saving)")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                seller_name = st.text_input(
                    "Seller Name",
                    value=suggestion or "",
                    help=(
                        f"ML: {ml_pred} ({ml_conf:.2f} confidence); Rules: {rule_pred}"
                        if ml_pred or rule_pred else ""
                    ),
                )
            with col_b:
                invoice_date = st.text_input(
                    "Invoice Date (YYYY-MM-DD or as on invoice)",
                    value=(llm_fields or {}).get("invoice_date", invoice_date),
                )
            with col_c:
                amount = st.number_input(
                    "Amount (Total incl. tax)",
                    min_value=0.0,
                    value=float((llm_fields or {}).get("amount", amount)),
                    step=0.01,
                )
            with col_d:
                invoice_number = st.text_input(
                    "Invoice Number",
                    value=(llm_fields or {}).get("invoice_number", invoice_number),
                )

            save_btn = st.form_submit_button("💾 Save to Database")

        if save_btn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO invoices (seller_name, invoice_date, amount, invoice_number) VALUES (?,?,?,?)",
                (seller_name, invoice_date, float(amount), invoice_number)
            )
            conn.commit()
            st.success(f"✅ Saved invoice for **{seller_name}** (₹{amount:,.2f})")
            # Save feedback for future training
            try:
                os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
                import json
                with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "text": ocr_text,
                        "seller": seller_name,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
    except Exception as e:
        st.error(f"Failed to process invoice: {e}")

# -------------------------------
# Dashboard Section
# -------------------------------
st.header("📈 Dashboard & Reports")

df = pd.read_sql_query("SELECT seller_name, invoice_date, amount, invoice_number FROM invoices ORDER BY id DESC", conn)

if df.empty:
    st.info("No invoices yet. Upload one to get started.")
    st.stop()

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Invoices", len(df))
with col2:
    st.metric("Unique Sellers", df['seller_name'].nunique())
with col3:
    st.metric("Total Amount (incl. tax)", f"₹ {df['amount'].sum():,.2f}")

# Normalize invoice_date to datetime where possible
df['invoice_date_parsed'] = pd.to_datetime(df['invoice_date'], errors='coerce')
df['month'] = df['invoice_date_parsed'].dt.to_period('M').astype(str)

# Filters
st.subheader("🔎 Filters")
colf1, colf2 = st.columns([2, 2])
with colf1:
    sellers = ["All"] + sorted(df['seller_name'].dropna().unique().tolist())
    seller_filter = st.selectbox("Seller", sellers)
with colf2:
    start_date = st.date_input("Start Date", value=None)
    end_date = st.date_input("End Date", value=None)

filtered = df.copy()
if seller_filter != "All":
    filtered = filtered[filtered['seller_name'] == seller_filter]

if start_date and end_date:
    mask = (filtered['invoice_date_parsed'] >= pd.to_datetime(start_date)) & \
           (filtered['invoice_date_parsed'] <= pd.to_datetime(end_date))
    filtered = filtered[mask]

st.subheader("📑 Filtered Invoices")

# Add bulk delete option
if not filtered.empty:
    col_bulk, col_info = st.columns([1, 3])
    with col_bulk:
        if st.button("🗑️ Delete All Filtered", type="secondary", help="Delete all invoices shown in current filter"):
            if st.session_state.get("confirm_bulk_delete", False):
                cur = conn.cursor()
                for idx, row in filtered.iterrows():
                    cur.execute("DELETE FROM invoices WHERE seller_name = ? AND invoice_date = ? AND amount = ? AND invoice_number = ?", 
                              (row['seller_name'], row['invoice_date'], row['amount'], row['invoice_number']))
                conn.commit()
                st.success(f"✅ Deleted {len(filtered)} invoices successfully!")
                st.rerun()
            else:
                st.session_state["confirm_bulk_delete"] = True
                st.warning(f"⚠️ Click again to confirm deletion of {len(filtered)} invoices")
                st.rerun()
    with col_info:
        st.info(f"Showing {len(filtered)} invoices. Use the 🗑️ button next to each invoice to delete individually.")

# Add delete functionality
    # Create a unique identifier for each row
    filtered['row_id'] = range(len(filtered))
    
    # Display data with delete buttons
    for idx, row in filtered.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
        with col1:
            st.write(f"**{row['seller_name']}**")
        with col2:
            st.write(row['invoice_date'])
        with col3:
            st.write(f"₹{row['amount']:,.2f}")
        with col4:
            st.write(row['invoice_number'] if row['invoice_number'] else "N/A")
        with col5:
            if st.button("🗑️", key=f"delete_{row['row_id']}", help="Delete this invoice"):
                # Add confirmation
                if st.session_state.get(f"confirm_delete_{row['row_id']}", False):
                    cur = conn.cursor()
                    cur.execute("DELETE FROM invoices WHERE seller_name = ? AND invoice_date = ? AND amount = ? AND invoice_number = ?", 
                              (row['seller_name'], row['invoice_date'], row['amount'], row['invoice_number']))
                    conn.commit()
                    st.success("✅ Invoice deleted successfully!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_{row['row_id']}"] = True
                    st.warning(f"⚠️ Click again to confirm deletion of invoice from {row['seller_name']}")
                    st.rerun()
        st.divider()
else:
    st.info("No invoices found with current filters.")

# Seller-wise summary
st.subheader("📌 Seller-wise Summary")
summary = filtered.groupby("seller_name", dropna=False).agg(
    total_invoices=("seller_name", "count"),
    total_amount=("amount", "sum")
).reset_index().sort_values("total_amount", ascending=False, ignore_index=True)
st.dataframe(summary)

# Charts
st.subheader("📊 Visual Insights")
if not summary.empty:
    fig_bar = px.bar(summary, x="seller_name", y="total_amount", text="total_amount",
                     title="Total Amount by Seller")
    st.plotly_chart(fig_bar, use_container_width=True)

# Monthly trend
monthly = filtered.dropna(subset=['month']).groupby('month', as_index=False).agg(total_amount=('amount','sum'))
if not monthly.empty:
    fig_line = px.line(monthly, x="month", y="total_amount", markers=True, title="Monthly Invoice Trend")
    st.plotly_chart(fig_line, use_container_width=True)

# Pie chart
if not summary.empty and summary['total_amount'].sum() > 0:
    fig_pie = px.pie(summary, names="seller_name", values="total_amount", hole=0.45,
                     title="Seller Contribution to Total Amount")
    st.plotly_chart(fig_pie, use_container_width=True)

# Export
st.subheader("⬇️ Export")
def to_excel_bytes(df_in: pd.DataFrame) -> bytes:
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df_in.to_excel(writer, index=False, sheet_name="Invoices")
        summary.to_excel(writer, index=False, sheet_name="Seller Summary")
        if not monthly.empty:
            monthly.to_excel(writer, index=False, sheet_name="Monthly Trend")
        # A cover sheet with company name
        pd.DataFrame({"Company":[st.session_state.company_name],
                      "Generated At":[datetime.now().strftime("%Y-%m-%d %H:%M")]}).to_excel(
            writer, index=False, sheet_name="About"
        )
    return buff.getvalue()

st.download_button(
    label="📥 Download Excel Report",
    data=to_excel_bytes(filtered[['seller_name','invoice_date','amount','invoice_number']]),
    file_name="invoice_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Tip: Improve OCR accuracy by uploading clear, high-resolution invoice images.")
