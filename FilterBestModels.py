import os
import shutil
import re

selected_models = {"densenet201": {1: [], 2: [], 3: []}, "resnet50": {1: [], 2: [], 3: []}}
total_models_found = 0
almost_selected_models = []

accuracy_threshold = 15.0  
min_improvement = 0.3     

layer_pattern = re.compile(r'_(\d+)layers_')
model_type_pattern = re.compile(r'(densenet201|resnet50)')

saved_models_folder = r"Git\Histopathological_Diagnosis\saved_models"  
destination_folder = r"Git\Histopathological_Diagnosis\best_models"  

os.makedirs(destination_folder, exist_ok=True)



with open(r"Git\Histopathological_Diagnosis\historial.txt", "r", encoding="utf-8") as file:
    for line in file:
        if "Configuration details:" in line:
            current_config = line
            accuracies = []  
            start_epoch = None
            end_epoch = None
            selected = False

        elif "Epoch" in line and "Train Accuracy:" in line:
            epoch_accuracy = float(line.split("Train Accuracy: ")[-1].replace("%", "").strip())
            accuracies.append(epoch_accuracy)  

            if start_epoch is None:
                start_epoch = epoch_accuracy
                selected = start_epoch < accuracy_threshold

            end_epoch = epoch_accuracy

        elif "Saved model:" in line:
            model_name = line.split("Saved model: ")[-1].strip()
            total_models_found += 1

            layer_match = layer_pattern.search(model_name)
            model_type_match = model_type_pattern.search(model_name)
            
            if layer_match and model_type_match:
                num_layers = int(layer_match.group(1))  
                model_type = model_type_match.group(1)  

                if selected and (end_epoch - start_epoch >= min_improvement):
                    if len(selected_models[model_type][num_layers]) < (3 if num_layers == 3 else 2):
                        selected_models[model_type][num_layers].append((model_name, accuracies))
                else:
                    almost_selected_models.append((model_name, accuracies, start_epoch, end_epoch, num_layers, model_type))

almost_selected_models.sort(key=lambda x: x[2])

for model_data in almost_selected_models:
    model_name, accuracies, start_epoch, end_epoch, num_layers, model_type = model_data
    
    if len(selected_models[model_type][num_layers]) < 5: #(3 if num_layers == 3 else 2):
        selected_models[model_type][num_layers].append((model_name, accuracies))


print(f"Total nº of models: {total_models_found}")
print(f"Best Models:")
for model_type, layers in selected_models.items():
    for num_layers, models in layers.items():
        print(f"{model_type} - {num_layers} layers: {len(models)} models")


for model_type in selected_models:
    print(f"\n{model_type.capitalize()}:")
    for num_layers in selected_models[model_type]:
        print(f"  {num_layers} layers:")
        for model_name, accuracies in selected_models[model_type][num_layers]:
            print(f"    - {model_name}: {accuracies}")


"""
"""
#ADAPT FORMAT OF MODELS FOR THE MAIN
"""
all_best_configurations = []
#regular expression to extract info of the keys 
pattern = re.compile(r'(?P<model_name>\w+?)_(?P<num_layers>\d+)layers_(?P<units>(?:\d+_?)+)_dropout(?P<dropout>[\d.]+)')

for model_file, accuracies in selected_models.items():
    match = pattern.search(model_file)
    if match:
        model_name = match.group("model_name")  #(ej., densenet201 o resnet50)
        num_layers = int(match.group("num_layers"))  #(ej., 3)
        
        layer_config = list(map(int, match.group("units").split('_')))
        
        dropout = float(match.group("dropout"))  #(ej., 0.25 o 0.5)

        configuration = {
            "model_name": model_name,
            "num_layers": num_layers,
            "units_per_layer": layer_config,
            "dropout": dropout
        }
        all_best_configurations.append(configuration)

for config in all_best_configurations:
    print(config)"""