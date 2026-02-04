import fitz  # PyMuPDF
import pytesseract
import numpy as np
import cv2
import re
from PIL import Image
from loguru import logger


# -------------------------------
# OCR Utilities
# -------------------------------

def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Improve image quality before OCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return thresh


def _ocr_page(page: fitz.Page) -> str:
    """
    Perform OCR on a PDF page using Tesseract.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    processed = _preprocess_image(img)
    text = pytesseract.image_to_string(processed)

    return text


# -------------------------------
# Hybrid Detection & Parsing
# -------------------------------

def _is_qa_page(text: str) -> bool:
    """
    Detect whether a page contains Q&A style content.
    """
    return bool(re.search(r"\bQ[.:]\s+.+?\?", text, re.IGNORECASE))


def _extract_qa_pairs(text: str) -> list[dict]:
    """
    Extract structured Q&A pairs from text.
    """
    qa_pairs = []

    pattern = re.compile(
        r"Q[.:]\s*(.+?)\s*Ans[.:]\s*(.+?)(?=\nQ[.:]|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(text)

    for question, answer in matches:
        qa_pairs.append({
            "question": question.strip(),
            "answer": re.sub(r"\s+", " ", answer.strip())
        })

    return qa_pairs


# -------------------------------
# Ingestion Service
# -------------------------------

class DocumentIngestionService:
    """
    Hybrid PDF ingestion service:
    - Full-text extraction
    - Selective OCR fallback
    - Structured Q&A extraction
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        logger.info(
            f"Ingestion initialized",
            extra={"pdf": pdf_path, "pages": len(self.doc)}
        )

    def ingest(self) -> dict:
        pages = []

        for page_number, page in enumerate(self.doc, start=1):
            logger.info(f"Processing page {page_number}")

            raw_text = page.get_text("text").strip()

            # OCR fallback if extracted text is weak
            if len(raw_text) < 50:
                logger.info(f"OCR triggered", extra={"page": page_number})
                raw_text = _ocr_page(page).strip()

            page_type = "qa" if _is_qa_page(raw_text) else "text"

            page_data = {
                "page": page_number,
                "type": page_type
            }

            if page_type == "qa":
                qa_pairs = _extract_qa_pairs(raw_text)

                # Hybrid safety fallback
                if qa_pairs:
                    page_data["qa_pairs"] = qa_pairs
                else:
                    page_data["type"] = "text"
                    page_data["content"] = raw_text
            else:
                page_data["content"] = raw_text

            pages.append(page_data)

        return {
            "document": self.pdf_path,
            "total_pages": len(pages),
            "pages": pages
        }
