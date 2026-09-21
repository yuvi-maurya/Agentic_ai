import re
from pathlib import Path

import chromadb

# --- Settings (ek jagah, taaki baad me badalna aasan ho) ---
BASE_DIR = Path(__file__).resolve().parent.parent   # project root (studybuddy/)
LIBRARY_DIR = BASE_DIR / "library"                  # jahan tumhare .txt notes hain
DB_DIR = BASE_DIR / "chroma_db"                     # jahan Chroma data save karega
COLLECTION_NAME = "study_notes"                     # Chroma ke andar "table" ka naam
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                # chhota local embedding model
MAX_CHARS = 500                                     # ek chunk ki max lambai (characters)

_model = None  # model yaha cache hoga, taaki baar-baar load na ho


def get_model():
    """Embedding model ko pehli baar zaroorat padne par hi load karta hai."""
    global _model
    if _model is None:
        # import yahin andar: taaki file import hote hi bhaari torch load na ho
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def chunk_text(text, max_chars=MAX_CHARS):
    """Text ko paragraphs me todta hai, phir chhote paragraphs ko jodkar chunks banata hai."""
    # blank line (\n\n) par split karo aur khaali hisse hata do
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        # agar current me p jodne se limit paar hoti hai, to current ko chunk bana do
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def embed(texts):
    """Texts ki list -> embeddings (har text ke liye numbers ki list)."""
    # normalize_embeddings=True: har vector ki lambai 1 kar deta hai, taaki distance fair rahe
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_client():
    """Disk par save hone wala Chroma database kholta hai."""
    return chromadb.PersistentClient(path=str(DB_DIR))


def ingest_library(library_dir=LIBRARY_DIR):
    """library ki saari .txt files ko chunk -> embed -> Chroma me store karta hai."""
    ids, documents, metadatas = [], [], []
    for file in sorted(Path(library_dir).glob("*.txt")):
        text = file.read_text(encoding="utf-8")   # Windows par utf-8 batana zaroori hai
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{file.stem}-{i}")                       # unique id, jaise water_cycle-0
            documents.append(chunk)                              # chunk ka asli text
            metadatas.append({"source": file.name, "chunk": i})  # ye chunk kis file se aaya

    if not documents:
        return 0

    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)   # purana data hatao (fresh rebuild)
    except Exception:
        pass                                        # pehli baar collection hai hi nahi, koi baat nahi
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed(documents),
    )
    return len(documents)


def retrieve(question, k=3):
    """Question ke sabse nazdeeki k chunks lautata hai (chhota distance = zyada match)."""
    collection = get_client().get_collection(COLLECTION_NAME)
    n = min(k, collection.count())   # DB me k se kam chunks hon to error na aaye
    result = collection.query(query_embeddings=embed([question]), n_results=n)
    hits = []
    # Chroma har cheez ki list-of-lists deta hai (har question ke liye ek list), isliye [0]
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits


if __name__ == "__main__":
    # Self-test: python study_agent/rag.py "apna sawal"
    import sys

    question = " ".join(sys.argv[1:]) or "What is evaporation?"
    print(f"Question: {question}\n")
    for hit in retrieve(question):
        print(f"[distance {hit['distance']:.3f}] {hit['source']}")
        print(hit["text"])
        print()