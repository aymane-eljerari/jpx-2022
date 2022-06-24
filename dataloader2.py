import os
import torch
import itertools
import pandas as pd
import numpy as np
from autoencoder import encoder
from torch.utils.data import Dataset
from dataloader import JPXData

class SecondData(Dataset):
    def __init__(self, encoder):
        data = pd.read_csv("modified_data/autoencoder_data.csv")
        dataset = JPXData(data, model="ae")
        latent_space = [encoder(i).flatten() for i in dataset.inputs]

        self.inputs = torch.stack(latent_space)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.inputs[idx]

if __name__ == "__main__":
    e = encoder([69, 50, 35, 20, 15]) 
    e.load_state_dict(torch.load("autoencoder_saved/ae1.pt"))
    e.eval()
    data = SecondData(e)
    print(data[0])
    print(data[0].shape)