import torch

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print(f"==> Saving checkpoint to: {filename}")
    torch.save(state, filename)
    return None

def load_checkpoint(filename, model, optimizer=None):
    print(f"==> Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval()
    return model, optimizer