import re
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def allowed_file(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in ALLOWED_EXTENSIONS


def extract_text_from_upload(file_storage) -> str:
    filename = file_storage.filename or ""
    extension = Path(filename).suffix.lower()
    content = file_storage.read()
    file_storage.stream.seek(0)

    if extension == ".pdf":
        return _extract_pdf_text(content)
    if extension == ".docx":
        return _extract_docx_text(content)
    if extension == ".txt":
        return content.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    extracted = []
    for page in reader.pages:
        extracted.append(page.extract_text() or "")
    return "\n".join(extracted).strip()


def _extract_docx_text(content: bytes) -> str:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    texts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            texts.append(node.text)
        elif node.tag.endswith("}p"):
            texts.append("\n")

    merged = " ".join(texts)
    merged = re.sub(r"\s+\n", "\n", merged)
    merged = re.sub(r"\n\s+", "\n", merged)
    merged = re.sub(r"\n{2,}", "\n\n", merged)
    return merged.strip()
