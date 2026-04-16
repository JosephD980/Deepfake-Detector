CONFIG = {
    "num_classes": 3,
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 10,
    "img_size": 224,
    "checkpoint_dir": "runs/",
    "device": "cuda",        
    "smoke_test": False,
    "max_samples": 80000,
    "dataset_sources": [
        "saberzl/So-Fake-Set",
        "saberzl/SID_Set",
    ]

}
