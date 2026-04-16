import os
import io
import base64
import gc          
import torch
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import get_model

# 1. FORCE PYTORCH TO USE 1 THREAD (CRITICAL)
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Real", 1: "Tampered", 2: "AI Generated"}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. PRE-LOAD MODEL AT STARTUP
def load_model():
    m = get_model()
    m.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

# Load this globally so it's ready before requests hit
GLOBAL_MODEL = load_model()

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
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        img_224 = img.resize((224, 224))
        raw_np = np.array(img_224).astype(np.float32) / 255.0
        tensor = transform(img_224).unsqueeze(0).to(DEVICE)

        # Probabilities
        with torch.no_grad():
            probs = torch.softmax(GLOBAL_MODEL(tensor), dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        # GradCAM
        with GradCAM(model=GLOBAL_MODEL, target_layers=[GLOBAL_MODEL.features[-1]]) as cam:
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(pred_idx)]
            )[0]

        visualization = show_cam_on_image(raw_np, grayscale_cam, use_rgb=True)

        buffered = io.BytesIO()
        Image.fromarray(visualization).save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        heatmap_url = f"data:image/png;base64,{heatmap_b64}"

        # 3. AGGRESSIVE MEMORY CLEANUP
        del tensor
        del grayscale_cam
        del visualization
        del img_224
        del raw_np
        gc.collect() # Force OS to reclaim RAM immediately

        return jsonify({
            "label": LABELS[pred_idx],
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                LABELS[i]: round(probs[i].item() * 100, 1) for i in range(3)
            },
            "heatmap_url": heatmap_url
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
