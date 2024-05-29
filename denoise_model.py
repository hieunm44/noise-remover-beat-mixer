import json
from network import CleanUNet
import torch
import torch.nn as nn

class DenoiseModel(nn.Module):
    def __init__(self, config_path, model_path, device):
        super(DenoiseModel, self).__init__()
        self.device = device
        with open(config_path) as f:
            data = f.read()
        self.config = json.loads(data)
        self.model = CleanUNet(**self.config["network_config"])
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
   
    
    def forward(self, noisy_audio):
        generated_audio = self.model(noisy_audio)
        return generated_audio
