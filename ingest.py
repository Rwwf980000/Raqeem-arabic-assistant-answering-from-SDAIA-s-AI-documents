"""Build the vector index: PDFs in data/ -> chunks -> bge-m3 embeddings -> Chroma.

Re-run it whenever you add or change documents; it rebuilds the index from scratch.
"""
import shutil
import unicodedata

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def clean(text: str) -> str:
    # NFKC turns Arabic "presentation form" glyphs that PDFs often contain
    # back into normal letters, so Arabic text embeds and matches properly.
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def load_pdfs():
    docs = []
    for pdf in sorted(config.DATA_DIR.rglob("*.pdf")):  # rglob also looks in subfolders
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.page_content = clean(p.page_content)
            p.metadata = {"source": pdf.name, "page": p.metadata.get("page", 0) + 1}
        pages = [p for p in pages if len(p.page_content) > 50]  # drop blank/cover pages
        print(f"{pdf.name}: {len(pages)} pages")
        docs.extend(pages)
    return docs


def build_index(embeddings=None):
    docs = load_pdfs()
    if not docs:
        raise SystemExit(f"No PDFs found in {config.DATA_DIR}. Check DATA_DIR in config.py.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "۔ ", "، ", " ", ""],  # includes Arabic punctuation
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks; embedding with {config.EMBED_MODEL}...")

    shutil.rmtree(config.DB_DIR, ignore_errors=True)
    Chroma.from_documents(
        chunks,
        embeddings or get_embeddings(),
        collection_name=config.COLLECTION,
        persist_directory=str(config.DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Index saved to {config.DB_DIR}")


if __name__ == "__main__":
    build_index()
