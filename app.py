import io, base64, os, sys
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import get_model

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FIX 3: Labels match index.html exactly
LABELS = {0: "Real", 1: "Tampered", 2: "AI Generated"}

# FIX 2: Added Resize so raw_np and CAM are both 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Lazy load — avoids gunicorn startup timeout on Render
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

        # FIX 2: Resize image so raw_np is 224x224, matching GradCAM output
        img_resized = img.resize((224, 224))
        raw_np = np.array(img_resized).astype(np.float32) / 255.0

        tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Get probabilities
        with torch.no_grad():
            out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item() * 100

        # Generate GradCAM — needs gradients, runs its own backward pass
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(pred)]
        )[0]

        visualization = show_cam_on_image(raw_np, grayscale_cam, use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(visualization).save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "label": LABELS[pred],
            "confidence": round(confidence, 1),
            "probabilities": {
                "Real":         round(probs[0].item() * 100, 1),
                "Tampered":     round(probs[1].item() * 100, 1),
                "AI Generated": round(probs[2].item() * 100, 1)
            },
            "heatmap": heatmap_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
