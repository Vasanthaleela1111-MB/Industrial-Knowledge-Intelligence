from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from email import policy
from email.parser import BytesParser
from pathlib import Path
import mimetypes
import subprocess
import tempfile
from datetime import datetime

import pandas as pd
from PIL import Image

from backend.config import DOCLING_TIMEOUT_SECONDS, USE_DOCLING

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover - keeps local demo alive if Docling extras fail.
    DocumentConverter = None

MIN_PDF_TEXT_CHARS = 200
MAX_OCR_PAGES = 50


class DoclingLoader:

    def __init__(self):
        self.converter = None

    def _get_converter(self):
        if not USE_DOCLING or not DocumentConverter:
            return None

        if self.converter is None:
            self.converter = DocumentConverter()

        return self.converter

    def load(self, file_path):
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)

        converter = self._get_converter()
        if converter:
            try:
                result = self._convert_with_docling(path, converter)
                if result:
                    return result
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"[Docling] Failed to parse {path.name}: {e}")

        return self._fallback_payload(path, suffix)

    def _load_pdf(self, path: Path) -> dict:
        fast_text = self._extract_pdf_text(path)
        if len(fast_text.strip()) >= MIN_PDF_TEXT_CHARS:
            print(
                f"[PDF] Extracted {len(fast_text)} characters "
                f"with pdftotext ({path.name})"
            )
            return {
                "text": fast_text,
                "document": None,
                "loader": "pdftotext",
            }

        ocr_text = self._extract_pdf_ocr(path)
        if len(ocr_text.strip()) >= MIN_PDF_TEXT_CHARS:
            print(
                f"[PDF] Extracted {len(ocr_text)} characters "
                f"with OCR ({path.name})"
            )
            return {
                "text": ocr_text,
                "document": None,
                "loader": "pdf-ocr",
            }

        converter = self._get_converter()
        if converter:
            try:
                result = self._convert_with_docling(path, converter)
                if result and result.get("text", "").strip():
                    return result
            except Exception as e:
                print(f"[Docling] PDF fallback failed for {path.name}: {e}")

        best_text = ocr_text.strip() or fast_text.strip()
        if best_text:
            return {
                "text": best_text,
                "document": None,
                "loader": "pdftotext" if best_text == fast_text.strip() else "pdf-ocr",
            }

        return self._fallback_payload(path, ".pdf")

    def _convert_with_docling(self, path: Path, converter) -> dict | None:
        print(f"[Docling] Processing {path.name} (timeout={DOCLING_TIMEOUT_SECONDS}s)...")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(converter.convert, str(path))
            try:
                result = future.result(timeout=DOCLING_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                print(
                    f"[Docling] Timed out after {DOCLING_TIMEOUT_SECONDS}s "
                    f"for {path.name}"
                )
                return None

        document = result.document
        markdown = document.export_to_markdown()
        print("=" * 60)
        print("DOCLING OUTPUT")
        print(markdown[:1000])
        print("=" * 60)
        markdown = markdown.replace("\f", "\n\n--- PAGE BREAK ---\n\n")
        if not markdown.strip():
            return None

        return {
            "text": markdown,
            "document": document,
            "loader": "docling",
        }

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            completed = subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    str(path),
                    "-",
                ],
                capture_output=True,
                check=True,
                timeout=120,
            )
            return completed.stdout.decode("utf-8", errors="ignore").strip()
        except FileNotFoundError:
            print("[PDF] pdftotext not found; install poppler-utils")
        except subprocess.TimeoutExpired:
            print(f"[PDF] pdftotext timed out for {path.name}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
            print(f"[PDF] pdftotext failed for {path.name}: {stderr or exc}")

        return ""

    def _extract_pdf_ocr(self, path: Path) -> str:
        if pytesseract is None:
            print("[PDF] pytesseract not available for OCR fallback")
            return ""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                prefix = str(Path(tmpdir) / "page")
                completed = subprocess.run(
                    [
                        "pdftoppm",
                        "-png",
                        "-r",
                        "200",
                        str(path),
                        prefix,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=300,
                )
                if completed.returncode != 0:
                    return ""

                pages = sorted(Path(tmpdir).glob("page-*.png"))
                texts = []
                for page in pages[:MAX_OCR_PAGES]:
                    with Image.open(page) as image:
                        text = pytesseract.image_to_string(image)
                    if text.strip():
                        texts.append(text.strip())

                return "\n\n".join(texts)
        except FileNotFoundError:
            print("[PDF] pdftoppm not found; install poppler-utils")
        except subprocess.TimeoutExpired:
            print(f"[PDF] OCR timed out for {path.name}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
            print(f"[PDF] OCR failed for {path.name}: {stderr or exc}")
        except Exception as exc:
            print(f"[PDF] OCR failed for {path.name}: {exc}")

        return ""

    def _fallback_payload(self, path: Path, suffix: str) -> dict:
        text = self._fallback_text(path, suffix)
        metadata = {
            "file_name": path.name,
            "file_type": suffix,
            "mime_type": mimetypes.guess_type(str(path))[0],
            "file_size": path.stat().st_size,
            "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        return {
            "text": text,
            "document": None,
            "loader": "fallback",
            "metadata": metadata,
        }

    def _fallback_text(self, path: Path, suffix: str) -> str:
        if suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(
                path,
                sep=separator,
                encoding_errors="ignore",
                low_memory=False,
            )

            summary = [
                f"CSV File: {path.name}",
                f"Rows: {len(frame)}",
                f"Columns: {len(frame.columns)}",
                "",
                "Columns:",
                ", ".join(frame.columns),
                "",
                "Sample Data:",
                frame.head(50).to_markdown(index=False),
            ]
            return "\n".join(summary)

        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None)
            return "\n\n".join(
                f"## Sheet: {name}\nRows: {len(sheet)}\nColumns: {len(sheet.columns)}\n\n{sheet.head(50).to_markdown(index=False)}"
                for name, sheet in sheets.items()
            )

        if suffix in {".eml"}:
            with path.open("rb") as handle:
                message = BytesParser(policy=policy.default).parse(handle)
            subject = message.get("subject", "")
            sender = message.get("from", "")
            body = message.get_body(preferencelist=("plain", "html"))
            date = message.get("date", "")
            to = message.get("to", "")

            return (
                f"Subject: {subject}\n"
                f"From: {sender}\n"
                f"To: {to}\n"
                f"Date: {date}\n\n"
                f"{body.get_content() if body else ''}"
            )

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).replace("\x00", "").strip()

        if not text:
            return "Empty document."

        return text.strip()
