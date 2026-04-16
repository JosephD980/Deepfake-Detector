import io, torch, os
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
from model import get_model
import io, torch, os, base64

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Real", 1: "Tampered", 2: "AI Generated"}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model = None

def get_loaded_model():
    global _model
    if _model is None:
        m = get_model()
        m.load_state_dict(torch.load("model.pt", map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m
    return _model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        model = get_loaded_model()

        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        img_224 = img.resize((224, 224))
        raw_np = np.array(img_224).astype(np.float32) / 255.0
        tensor = transform(img_224).unsqueeze(0).to(DEVICE)

        # Probabilities
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        # GradCAM
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]
        visualization = show_cam_on_image(raw_np, grayscale_cam, use_rgb=True)

        buffered = io.BytesIO()
        Image.fromarray(visualization).save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "label": LABELS[pred_idx],
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                LABELS[i]: round(probs[i].item() * 100, 1) for i in range(3)
            },
            "heatmap_url": f"data:image/png;base64,{heatmap_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
