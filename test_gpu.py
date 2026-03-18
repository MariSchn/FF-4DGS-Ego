import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    # Prova a creare un tensore (qui è dove prima crashava)
    try:
        x = torch.ones((1, 1), device="cuda")
        print("Success: Tensor created on GPU!")
    except Exception as e:
        print(f"Error: {e}")