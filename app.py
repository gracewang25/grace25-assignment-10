from flask import Flask, render_template, request
from clip_model import load_image_embedding, load_text_embedding, hybrid_query
from search import find_similar_images
from pca_model import load_pca_model, apply_pca
import os

app = Flask(__name__)

# Load Embeddings Paths
IMAGE_EMBEDDINGS_PATH = "data/image_embeddings.pickle"
PCA_EMBEDDINGS_PATH = "data/pca_embeddings.pickle"
IMAGES_DIR = "static/coco_images_resized"

# Load PCA model
pca_model = load_pca_model(IMAGE_EMBEDDINGS_PATH, PCA_EMBEDDINGS_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        # Collect form data
        text_query = request.form.get("text_query")
        uploaded_file = request.files.get("image_query")
        weight = float(request.form.get("weight", 0.5))
        query_type = request.form.get("query_type")
        use_pca = request.form.get("use_pca") == "on"
        k = int(request.form.get("k", 50))

        query_embedding = None

        if query_type == "Text query" and text_query:
            query_embedding = load_text_embedding(text_query)

        elif query_type == "Image query" and uploaded_file:
            query_embedding = load_image_embedding(uploaded_file)
            if use_pca:
                query_embedding = apply_pca(pca_model, query_embedding, k)

        elif query_type == "Hybrid query" and text_query and uploaded_file:
            text_embedding = load_text_embedding(text_query)
            image_embedding = load_image_embedding(uploaded_file)
            
            if use_pca:
                image_embedding = apply_pca(pca_model, image_embedding, k, target_dim=512)
            
            query_embedding = hybrid_query(text_embedding, image_embedding, weight)
            
        # Perform the search if valid query
        if query_embedding is not None:
            results = find_similar_images(
                query_embedding,
                PCA_EMBEDDINGS_PATH if use_pca else IMAGE_EMBEDDINGS_PATH,
                IMAGES_DIR,
            )

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)