import torch
print(torch.__version__)


epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

for it, epoch in enumerate(epochs):
    print(it, epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("working over: {}".format(device))

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")

print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")