"""Step 1 of the knowledge pipeline: brand files (.pdf / .txt / .md) -> Documents."""
import re
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = {".pdf", ".txt", ".md"}


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_file(path: str, display_name: str | None = None) -> list[Document]:
    """Return one Document per PDF page, or one Document for a text/markdown file."""
    p = Path(path)
    name = display_name or p.name
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = []
        for page in PyPDFLoader(str(p)).load():
            text = clean_pdf_text(page.page_content)
            if text:
                docs.append(Document(page_content=text,
                                     metadata={"source_name": name, "page": page.metadata.get("page", 0)}))
        return docs
    if suffix in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        return [Document(page_content=text, metadata={"source_name": name, "page": 0})] if text else []
    raise ValueError(f"Unsupported file type: {suffix}")
