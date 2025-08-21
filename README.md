
# 🏭 Invoice Management Dashboard (Local, OCR-powered)

This is a **local Streamlit app** for uploading invoice images (JPG/PNG), extracting key fields via **OCR**, 
and storing them in a local **SQLite database**. It includes a professional dashboard with filters, charts, 
and an Excel export. Company name is configurable in the sidebar.

## ✨ Features
- Upload invoice images → auto-extract **Seller, Invoice Date, Amount**
- Store to **SQLite** (local file `invoices.db`)
- **Filters**: seller + date range
- **Charts**: seller-wise bar chart, monthly trend, seller contribution pie
- **Export**: one-click Excel with multiple sheets
- **Branding**: add your own logo in `assets/logo.png` and set company name in the app

## 🧰 Tech Stack
- Python, Streamlit
- SQLite (file-based DB)
- OCR via **pytesseract** (requires Tesseract engine installed)
- Plotly for charts

## ⚙️ Prerequisites
1. **Python 3.10+** installed
2. **Tesseract OCR engine** installed on your system:
   - **macOS (Homebrew):**
     ```bash
     brew install tesseract
     ```
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get update && sudo apt-get install -y tesseract-ocr
     ```
   - **Windows:**
     - Download installer: https://github.com/UB-Mannheim/tesseract/wiki
     - After install, ensure `tesseract.exe` is in PATH or set it in code:
       ```python
       import pytesseract
       pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
       ```

## 🚀 Run Locally
```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

The app will open in your browser (usually at http://localhost:8501).

## 🤖 Train the Seller Model
You can train a simple text classifier that predicts the seller from OCR text.

Option A: From the app sidebar
- Open the app and click "Train/Resume Seller Model" in the sidebar. It will train on the `DATASET/` folder and save to `models/seller_model.joblib`. Clicking again will resume training on the same data (keeps learning).

Option B: From terminal
```bash
# Train for 5 epochs and resume from any existing model
python train_model.py --root DATASET --out models/seller_model.joblib --epochs 5 --resume

# Quick smoke test on a subset
python train_model.py --root DATASET --limit 100 --epochs 2 --resume
```

Notes:
- The model is incremental (SGD). Using `--resume` continues learning from the saved model, so rerunning on the same data improves the fit over multiple passes.
- To reset training from scratch, delete `models/seller_model.joblib` and run training again (without `--resume`).
- The app will use the trained model automatically for seller prediction, with a rule-based fallback if the model is missing.

## 🆓 Optional: Use a local LLM (Ollama) for better extraction
You can enable a free, local LLM to parse fields more accurately. This is optional and privacy-preserving.

1) Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
# macOS users can also use Homebrew: brew install ollama
```

2) Pull a small model
```bash
ollama pull llama3.2:3b
```

3) Run Ollama
```bash
ollama serve
# Ollama listens at http://localhost:11434 by default
```

4) Enable in the app
- In the sidebar, check "Use local LLM (Ollama) for extraction" and keep the model as `llama3.2:3b` (or change if you pulled a different one).
- The LLM will attempt to extract `seller_name`, `invoice_date`, `amount`, and `invoice_number` from OCR text. You can still edit values before saving.

## 🖼️ Branding (Optional)
- Replace `assets/logo.png` with your company logo (any PNG).
- Set your **Company Name** from the app sidebar; it will appear in the UI and inside exported Excel.

## 🗂️ Project Structure
```
invoice_dashboard/
├─ app.py
├─ invoices.db            # created automatically at runtime
├─ requirements.txt
├─ README.md
└─ assets/
   └─ logo.png            # replace with your logo (optional)
```

## 🔐 Data
- Stored locally in `invoices.db` (SQLite). 
- To back up: copy this file. To reset: delete it (you'll lose records).

## 🧪 Notes on OCR Accuracy
- Use sharp, high-resolution images.
- If dates/amounts parse incorrectly, edit values before saving.
- You can improve parsing with vendor-specific rules or add PDF support later.

---

Made for manufacturing businesses that want a **simple, local, and private** invoice tracker.
