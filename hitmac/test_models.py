import torch

# Load the .pth file
model_weights = torch.load("diayn_executor_library_mate_v2.pth", map_location="cpu")

# # Check what it contains
# print(type(model_weights))  # usually dict or OrderedDict
# print(model_weights.keys()) # shows the layers/parameters

# # Example: inspect a specific weight tensor
# print(len(model_weights))

print(model_weights["executors"][0].keys())
