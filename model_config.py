"""# Define model configurations
    model_options = ["resnet50", "densenet201"]
    num_layers_options = range(1, 4)
    units_per_layer_options = [64, 128, 256]
    dropout_options = [0.25]
    all_configurations = []

    # Generate all configurations
    for num_layers in num_layers_options:
        layer_configs = list(product(units_per_layer_options, repeat=num_layers))
        if num_layers == 2:
            layer_configs = [config for config in layer_configs if len(set(config)) == 2]
        elif num_layers == 3:
            layer_configs = [config for config in layer_configs if len(set(config)) > 1]
        for model_name, layer_config, dropout in product(model_options, layer_configs, dropout_options):
            configuration = {
                "model_name": model_name,
                "num_layers": num_layers,
                "units_per_layer": list(layer_config),
                "dropout": dropout
            }
            all_configurations.append(configuration)"""

all_configurations = [
    #1 layer: resnet50
    {
        "model_name": "resnet50",
        "num_layers": 1,
        "units_per_layer": [128],
        "dropout": 0.25
    },
    {
        "model_name": "resnet50",
        "num_layers": 1,
        "units_per_layer": [256],
        "dropout": 0.25
    },
    #1 layer: densenet201
    {
        "model_name": "densenet201",
        "num_layers": 1,
        "units_per_layer": [128],
        "dropout": 0.25
    },
    {
        "model_name": "densenet201",
        "num_layers": 1,
        "units_per_layer": [256],
        "dropout": 0.25
    },
    #2 layers: resnet50
    {
        "model_name": "resnet50",
        "num_layers": 2,
        "units_per_layer": [128, 256],
        "dropout": 0.25
    },
    {
        "model_name": "resnet50",
        "num_layers": 2,
        "units_per_layer": [64, 128],
        "dropout": 0.25
    },
    #2 layers: densenet201
    {
        "model_name": "densenet201",
        "num_layers": 2,
        "units_per_layer": [128, 256],
        "dropout": 0.25
    },
    {
        "model_name": "densenet201",
        "num_layers": 2,
        "units_per_layer": [64, 128],
        "dropout": 0.25
    },]
"""
    #3 layers: resnet50
    {
        "model_name": "resnet50",
        "num_layers": 3,
        "units_per_layer": [64, 256, 64],
        "dropout": 0.25
    },
    {
        "model_name": "resnet50",
        "num_layers": 3,
        "units_per_layer": [128, 64, 256],
        "dropout": 0.25
    },
    #3 layers: densenet201
    {
        "model_name": "densenet201",
        "num_layers": 3,
        "units_per_layer": [64, 256, 64],
        "dropout": 0.25
    },
    {
        "model_name": "densenet201",
        "num_layers": 3,
        "units_per_layer": [128, 64, 256],
        "dropout": 0.25
    }
]
"""