import numpy as np
import pickle
from sklearn.decomposition import PCA


def load_pca_model(embeddings_path, pca_embeddings_path, n_components=50, target_dim=512):
    try:
        # Load original CLIP embeddings
        with open(embeddings_path, "rb") as f:
            embeddings_df = pickle.load(f)
            embeddings = np.vstack(embeddings_df["embedding"])

        # Train PCA model
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Pad reduced embeddings to match target_dim
        if reduced_embeddings.shape[1] < target_dim:
            padding = np.zeros((reduced_embeddings.shape[0], target_dim - reduced_embeddings.shape[1]))
            padded_embeddings = np.hstack((reduced_embeddings, padding))
        else:
            padded_embeddings = reduced_embeddings

        # Save PCA embeddings using pickle
        with open(pca_embeddings_path, "wb") as f:
            pickle.dump(
                {"file_name": embeddings_df["file_name"], "embedding": padded_embeddings}, f
            )
        print(f"PCA embeddings successfully saved to {pca_embeddings_path}")

        return pca
    except Exception as e:
        print(f"Error generating PCA embeddings: {e}")
        raise


def apply_pca(pca_model, embedding, n_components=50, target_dim=512):
    """
    Applies PCA to a single query embedding and pads to target_dim if needed.
    """
    reduced_embedding = pca_model.transform([embedding])[:, :n_components]
    
    # Pad to target_dim if needed
    if reduced_embedding.shape[1] < target_dim:
        padding = np.zeros((1, target_dim - reduced_embedding.shape[1]))
        reduced_embedding = np.hstack((reduced_embedding, padding))
    
    return reduced_embedding.flatten()