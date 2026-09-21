from rag import DB_DIR, LIBRARY_DIR, ingest_library

count = ingest_library()
if count == 0:
    print(f"Koi .txt file nahi mili: {LIBRARY_DIR}")
else:
    print(f"Done: {count} chunks store hue -> {DB_DIR}")