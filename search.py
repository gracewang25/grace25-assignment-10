import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_images(query_embedding, embeddings_path, images_dir, top_k=5):
    try:
        # Load precomputed embeddings using pickle
        with open(embeddings_path, "rb") as f:
            embeddings_df = pickle.load(f)

        embeddings = np.vstack(embeddings_df["embedding"])
        file_names = embeddings_df["file_name"]

        # Ensure matching shapes
        if query_embedding.shape[0] != embeddings.shape[1]:
            raise ValueError(
                f"Incompatible dimensions: Query shape {query_embedding.shape} vs "
                f"Embeddings shape {embeddings.shape[1]}"
            )

        # Compute cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Retrieve results
        results = [
            (f"{images_dir}/{file_names[idx]}", similarities[idx])
            for idx in top_indices
        ]
        return results
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []