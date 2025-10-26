import os
from torchvision import transforms
from Models.CNN_Encoder import VGG16
import numpy as np
import torch

def extract_visual_features(frames_path: str, features_path: str):
    """Extracts the features from frames using the pretrained VGG 16 network"""
    files = os.listdir(frames_path)
    
    # A standard transform needed to be applied to inputs of the models pre-trained on ImageNet
    transform_ = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cnn_encoder = VGG16()
    device = 'cpu'
    #device = 'cuda:0'
    cnn_encoder.to(device)
    
    for file in files:
        print('Extracting features of %s' % file)
        frames = np.load(os.path.join(frames_path, file))
        frames_tensor = torch.cat([transform_(frame).unsqueeze(dim=0) for frame in frames], dim=0)
        features = cnn_encoder(frames_tensor.to(device))
        out_file = os.path.join(features_path, file.replace('.npy', '_features.pt'))
        torch.save(features, out_file)