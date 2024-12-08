from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import torch
import torch.nn.functional as F

# Correct model name
MODEL_NAME = "ViT-B-32"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(MODEL_NAME, pretrained="openai")
model = model.to(device)
model.eval()

# Initialize tokenizer
tokenizer = get_tokenizer(MODEL_NAME)

def load_image_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = F.normalize(model.encode_image(image_tensor)).cpu().numpy()
    return embedding.flatten()

def load_text_embedding(text_query):
    tokens = tokenizer([text_query]).to(device)  # Correct tokenizer usage
    with torch.no_grad():
        embedding = F.normalize(model.encode_text(tokens)).cpu().numpy()
    return embedding.flatten()

def hybrid_query(text_embedding, image_embedding, weight=0.5):
    """
    Combines text and image embeddings with a weighted average.
    """
    text_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(device)
    image_tensor = torch.tensor(image_embedding, dtype=torch.float32).to(device)

    # Ensure matching shapes
    if text_tensor.shape != image_tensor.shape:
        raise ValueError(
            f"Shape mismatch: text_tensor {text_tensor.shape} vs image_tensor {image_tensor.shape}"
        )

    # Perform weighted combination and normalization
    combined_embedding = weight * text_tensor + (1 - weight) * image_tensor
    return torch.nn.functional.normalize(combined_embedding, p=2, dim=0).cpu().numpy()