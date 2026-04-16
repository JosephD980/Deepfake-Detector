import torch.nn as nn
from torchvision import models
from config import CONFIG

def get_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        CONFIG["num_classes"]
    )
    return model
