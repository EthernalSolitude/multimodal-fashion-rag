import os

from datasets import load_dataset
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config import settings

IMAGES_DIR = "./images"
os.makedirs(IMAGES_DIR, exist_ok=True)

print("Multimodal RAG: Qdrant + CLIP (dense) + BM42 (sparse)")

client = QdrantClient(url=settings.qdrant_url)
print(f" Qdrant: {settings.qdrant_url}")

dense_model = SentenceTransformer('./models/clip-multilingual')
print(" CLIP dense (512)")

sparse_model = SparseTextEmbedding(settings.sparse_model_name)
print(f" Sparse: {settings.sparse_model_name}")

dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:5000]")
print(f" {len(dataset)} товаров")

client.recreate_collection(
    collection_name="fashion",
    vectors_config={"dense": VectorParams(size=512, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)

BATCH_SIZE = 64
points = []
total = 0

texts_batch = []
rows_batch = []


def flush(points_buf):
    if points_buf:
        client.upsert("fashion", points_buf)


def build_points(rows, texts):
    dense_vecs = dense_model.encode(texts, batch_size=32, show_progress_bar=False)
    sparse_vecs = list(sparse_model.embed(texts))
    out = []
    for row, dvec, svec in zip(rows, dense_vecs, sparse_vecs):
        image_path = f"{IMAGES_DIR}/{row['id']}.jpg"
        if not os.path.exists(image_path):
            row['image'].save(image_path)
        out.append(PointStruct(
            id=int(row['id']),
            vector={
                "dense": dvec.tolist(),
                "sparse": SparseVector(
                    indices=svec.indices.tolist(),
                    values=svec.values.tolist(),
                ),
            },
            payload={
                "title": row['productDisplayName'],
                "category": row['articleType'],
                "gender": row['gender'],
                "color": row['baseColour'] or "unknown",
                "product_id": int(row['id']),
                "image_path": image_path,
            },
        ))
    return out


print(" Индексируем...")
for row in dataset:
    try:
        text = f"{row['productDisplayName']} {row['articleType']} {row['gender']} {row['baseColour']}"
        texts_batch.append(text)
        rows_batch.append(row)
        if len(texts_batch) >= BATCH_SIZE:
            pts = build_points(rows_batch, texts_batch)
            flush(pts)
            total += len(pts)
            texts_batch, rows_batch = [], []
            print(f"  {total}/5000")
    except Exception as e:
        print(f"Пропуск: {e}")
        continue

if texts_batch:
    pts = build_points(rows_batch, texts_batch)
    flush(pts)
    total += len(pts)

print(f"{total} товаров в векторной БД")
