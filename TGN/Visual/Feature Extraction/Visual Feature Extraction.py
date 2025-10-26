import os
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = vgg16(pretrained=True, progress=True)
        features = list(self.model.classifier.children())[:-1]  # removing the last layer
        self.model.classifier = nn.Sequential(*features)

        # freeze the weights of the network so it will not be trained
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        :param x: A bunch of frames as a torch.Tensor with shape (K, 224, 224, 3)
        :return: a torch.Tensor containing the extracted features with shape (K, 4096)
        """
        self.model.eval()
        return self.model(x)
        #return torch.zeros([x.shape[0], 4096])


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