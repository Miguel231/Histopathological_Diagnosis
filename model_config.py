# model_configs.py

all_configurations = [
    {
        "model_name": "resnet50",
        "num_layers": 1,
        "units_per_layer": [128],
        "dropout": 0.25
    },
    {
        "model_name": "densenet201",
        "num_layers": 2,
        "units_per_layer": [64, 256],
        "dropout": 0.25
    },
    {
        "model_name": "resnet50",
        "num_layers": 2,
        "units_per_layer": [128, 128],
        "dropout": 0.25
    },
    {
        "model_name": "densenet201",
        "num_layers": 3,
        "units_per_layer": [64, 128, 256],
        "dropout": 0.25
    },
    {
        "model_name": "resnet50",
        "num_layers": 3,
        "units_per_layer": [256, 128, 64],
        "dropout": 0.25
    }
]
