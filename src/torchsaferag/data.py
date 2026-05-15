from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any
import itertools

from pypdf import PdfReader
from bs4 import BeautifulSoup


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_fixed(text: str, max_tokens: int = 400, overlap: int = 40) -> List[str]:
    """Simple whitespace token splitter; token ~= word for robustness without tokenizer."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def split_sentence_blocks(text: str, max_tokens: int = 220, overlap: int = 20) -> List[str]:
    """
    Sentence-aware chunking: split on punctuation, pack sentences until max_tokens words.
    Uses simple regex sentence split to avoid heavy dependencies.
    """
    sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", text)
    blocks = []
    buffer = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if word_count + len(words) > max_tokens and buffer:
            blocks.append(" ".join(buffer))
            # apply overlap: carry last 'overlap' words to next buffer
            carry = list(itertools.chain.from_iterable(b.split() for b in buffer))[-overlap:] if overlap > 0 else []
            buffer = [" ".join(carry)] if carry else []
            word_count = len(carry)
        buffer.append(sent)
        word_count += len(words)
    if buffer:
        blocks.append(" ".join(buffer))
    return blocks


def extract_html_sections(text: str) -> List[Dict[str, Any]]:
    """
    Parse HTML, return list of dicts with section title and paragraph text.
    """
    soup = BeautifulSoup(text, "lxml")
    sections = []
    current_title = ""
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ["h1", "h2", "h3"]:
            current_title = el.get_text(" ", strip=True)
        else:
            para = el.get_text(" ", strip=True)
            if para:
                sections.append({"section": current_title, "text": para})
    return sections


def keywords_from_text(text: str, top_k: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z_\.]{3,}", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def ingest_paths(
    input_paths: Iterable[Path],
    output_path: Path,
    max_tokens: int = 400,
    overlap: int = 40,
    source_prefix: str = "",
    method: str = "fixed",
) -> None:
    """
    Read files, normalize, chunk, and write JSONL with fields: id, text, metadata.
    Supports .txt, .md, .html/.htm, .pdf.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counter = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in input_paths:
            ext = path.suffix.lower()
            if ext in {".txt", ".md", ".html", ".htm"}:
                raw = read_text_file(path)
            elif ext == ".pdf":
                raw = read_pdf(path)
            else:
                continue

            if method == "structured" and ext in {".html", ".htm"}:
                sections = extract_html_sections(raw)
                for sec_idx, sec in enumerate(sections):
                    cleaned = normalize_text(sec["text"])
                    parts = split_sentence_blocks(cleaned, max_tokens=max_tokens, overlap=overlap)
                    for chunk_idx, chunk_text in enumerate(parts):
                        chunk_id = f"{source_prefix}{path.stem}_s{sec_idx}_{chunk_idx}"
                        meta = {
                            "source": str(path),
                            "section": sec.get("section", ""),
                            "page": None,
                            "chunk_index": chunk_idx,
                            "keywords": keywords_from_text(chunk_text),
                        }
                        chunk = Chunk(id=chunk_id, text=chunk_text, metadata=meta)
                        out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                        counter += 1
            elif method == "structured" and ext == ".pdf":
                reader = PdfReader(path)
                for pageno, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    cleaned_page = normalize_text(page_text)
                    parts = split_sentence_blocks(cleaned_page, max_tokens=max_tokens, overlap=overlap)
                    for chunk_idx, chunk_text in enumerate(parts):
                        chunk_id = f"{source_prefix}{path.stem}_p{pageno}_{chunk_idx}"
                        meta = {
                            "source": str(path),
                            "section": None,
                            "page": pageno,
                            "chunk_index": chunk_idx,
                            "keywords": keywords_from_text(chunk_text),
                        }
                        chunk = Chunk(id=chunk_id, text=chunk_text, metadata=meta)
                        out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                        counter += 1
            else:
                cleaned = normalize_text(raw)
                if method == "sentence":
                    parts = split_sentence_blocks(cleaned, max_tokens=max_tokens, overlap=overlap)
                else:
                    parts = split_fixed(cleaned, max_tokens=max_tokens, overlap=overlap)
                for chunk_idx, chunk_text in enumerate(parts):
                    chunk_id = f"{source_prefix}{path.stem}_{chunk_idx}"
                    meta = {
                        "source": str(path),
                        "section": None,
                        "page": None,
                        "chunk_index": chunk_idx,
                        "keywords": keywords_from_text(chunk_text),
                    }
                    chunk = Chunk(id=chunk_id, text=chunk_text, metadata=meta)
                    out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    counter += 1
    print(f"Wrote {counter} chunks to {output_path}")
