import io, base64, torch, cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model import get_model

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Real", 1: "Tampered", 2: "AI Generated"}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ── Lazy model loading — avoids startup timeout on Render ────────────────────
_model = None
_grad_cam = None

def get_loaded_model():
    global _model, _grad_cam
    if _model is None:
        m = get_model()
        m.load_state_dict(torch.load("model.pt", map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m
        _grad_cam = GradCAM(_model, _model.features[-1])
    return _model, _grad_cam

# ── Heatmap overlay ───────────────────────────────────────────────────────────
def overlay_heatmap(pil_img, cam):
    img_np = np.array(pil_img.resize((224, 224))).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        model, grad_cam = get_loaded_model()

        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # Probabilities — no grad needed
        with torch.no_grad():
            output = model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        # Grad-CAM — needs its own forward+backward pass
        cam = grad_cam.generate(tensor.clone(), pred_idx)
        heatmap_b64 = overlay_heatmap(img, cam)

        return jsonify({
            "label": LABELS[pred_idx],
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                LABELS[i]: round(probs[i].item() * 100, 1) for i in range(3)
            },
            "heatmap": heatmap_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
