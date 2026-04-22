import torch

def get_dummy_meta():
    """Returns dummy metadata list [party_enc, state_enc, barely_true, false, half_true, mostly_true, pants_on_fire]"""
    return [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]

def get_meta_tensor(meta_list, device='cpu'):
    """Converts the list to a correctly shaped tensor for the model"""
    return torch.tensor([meta_list], dtype=torch.float).to(device)
