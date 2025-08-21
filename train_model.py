import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
from PIL import Image
import pytesseract
import pdfplumber
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# Keep parsing consistent with app.py/import_dataset.py where possible
DATE_PATTERNS = [
    r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
    r"\b\d{1,2}\s*[-]\s*[A-Za-z]{3,9}\s*[-]\s*\d{4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
    r"(?:dated|date)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
]

def extract_text_from_pdf_path(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join((page.extract_text() or "") + "\n" for page in pdf.pages)
            if text.strip():
                return text.strip()
        with pdf_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)
            return text.strip()
    except Exception:
        return ""

def ocr_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png"]:
        image = Image.open(path).convert("RGB")
        return pytesseract.image_to_string(image)
    if suffix == ".pdf":
        return extract_text_from_pdf_path(path)
    return ""

def guess_seller_name(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    specific_companies = [
        "Tradecon International",
        "Jenendra Press",
        "Shree Jenendra",
        "Jenendra",
    ]
    for line in lines:
        for company in specific_companies:
            if company.lower() in line.lower():
                return company

    for i, line in enumerate(lines):
        if re.search(r"tax\s+invoice", line, flags=re.IGNORECASE):
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line or re.search(r"(invoice|gstin|address|state|code|terms|payment|delivery|note|mode)", next_line, flags=re.IGNORECASE):
                    continue
                if re.search(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", next_line) and len(next_line) < 50:
                    return next_line

    for line in lines[:10]:
        if re.search(r"(invoice|tax|gst|bill|receipt|date|amount|total|due|paid|buyer|consignee|terms|payment|way|delivery|note|mode|contact|no|ack|state|name)", line, flags=re.IGNORECASE):
            continue
        if line.isupper():
            continue
        if re.match(r"^[\d\s\-\.\/]+$", line):
            continue
        if len(line) > 50:
            continue
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", line):
            return line

    return "Unknown Seller"

def collect_examples(dataset_root: Path) -> List[Tuple[str, str]]:
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.pdf"):
        paths.extend(dataset_root.rglob(ext))
    paths.sort()

    examples: List[Tuple[str, str]] = []
    for p in paths:
        try:
            text = ocr_text_from_path(p)
            if not text.strip():
                continue
            seller = guess_seller_name(text)
            if not seller:
                seller = "Unknown Seller"
            examples.append((text, seller))
        except Exception:
            continue
    return examples

def load_feedback(feedback_path: Path) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    if not feedback_path.exists():
        return examples
    try:
        with feedback_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                seller = obj.get("seller", "")
                if text and seller:
                    examples.append((text, seller))
    except Exception:
        pass
    return examples

def build_and_train(
    examples: List[Tuple[str, str]],
    model_path: Path,
    epochs: int = 3,
    resume: bool = False,
) -> None:
    texts = [t for t, _ in examples]
    labels = [y for _, y in examples]

    tfidf: TfidfVectorizer
    clf: SGDClassifier
    classes = sorted(list(set(labels)))

    if resume and model_path.exists():
        loaded = joblib.load(model_path)
        tfidf = loaded["tfidf"]
        clf = loaded["clf"]
        X_all = tfidf.transform(texts)
    else:
        tfidf = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )
        X_all = tfidf.fit_transform(texts)
        clf = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            learning_rate="optimal",
            tol=None,
            random_state=42,
        )

    # Train-test split for reporting only; disable stratify if any class has <2 samples
    from collections import Counter
    label_counts = Counter(labels)
    can_stratify = len(set(labels)) > 1 and all(c >= 2 for c in label_counts.values())
    can_split = len(labels) >= 4 and can_stratify
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, labels, test_size=0.15, random_state=42, stratify=labels
        )
    else:
        X_train, y_train = X_all, labels
        X_test, y_test = None, None

    # Multi-epoch training via partial_fit
    for epoch in range(max(1, epochs)):
        clf.partial_fit(X_train, y_train, classes=classes if epoch == 0 and not (resume and model_path.exists()) else None)

    if X_test is not None:
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tfidf": tfidf, "clf": clf, "classes": classes}, model_path)
    print(f"Saved model to {model_path}")

def train_seller_model(dataset_root: str, out_path: str, limit: int | None, epochs: int, resume: bool, feedback_path: str | None = None) -> None:
    dataset_root_path = Path(dataset_root)
    model_path = Path(out_path)
    if not dataset_root_path.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root_path}")

    examples = collect_examples(dataset_root_path)
    if limit is not None:
        examples = examples[: limit]

    # Append feedback examples
    if feedback_path:
        examples.extend(load_feedback(Path(feedback_path)))

    if not examples:
        raise SystemExit("No training examples found. Ensure dataset contains images/PDFs.")

    print(f"Loaded {len(examples)} examples. Training for {epochs} epoch(s)...")
    build_and_train(examples, model_path, epochs=epochs, resume=resume)

def main():
    parser = argparse.ArgumentParser(description="Train seller classifier from invoice OCR text")
    parser.add_argument("--root", type=str, default=str(Path(__file__).parent / "DATASET"), help="Dataset root containing images/PDFs")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "models" / "seller_model.joblib"), help="Output model path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of files to process (for quick runs)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs over the dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing model if present")
    parser.add_argument("--feedback", type=str, default=str(Path(__file__).parent / "feedback" / "training_samples.jsonl"), help="Path to JSONL feedback file to include in training")
    args = parser.parse_args()

    train_seller_model(args.root, args.out, args.limit, args.epochs, args.resume, args.feedback)

if __name__ == "__main__":
    main()


