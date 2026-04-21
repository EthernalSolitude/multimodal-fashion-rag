from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder, SentenceTransformer

print("Скачиваем clip-ViT-B-32-multilingual-v1...")
SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1').save('./models/clip-multilingual')

print("Скачиваем clip-ViT-B-32...")
SentenceTransformer('clip-ViT-B-32').save('./models/clip-vit-b32')

print("Скачиваем cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (мультиязычный reranker)...")
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1').save('./models/reranker')

print("Скачиваем BM42 sparse (Qdrant/bm42-all-minilm-l6-v2-attentions)...")
SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

print("Готово! Модели сохранены в ./models/")
