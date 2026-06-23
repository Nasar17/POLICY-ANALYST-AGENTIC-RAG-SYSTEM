from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-small-en-v1.5")
vec = m.encode("public records request deadline")
print(vec.shape)
