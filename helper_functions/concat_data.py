import pandas as pd
import os

def concat_data(train_path = "data/train_files/", supplemental_path= "data/supplemental_files/"):
    """
    Concatenate Train and Supplemental
    Data into a signle file
    """
    
    files =  os.listdir(train_path)

    for i in files:

        train           = pd.read_csv(train_path + i)
        supplemental    = pd.read_csv(supplemental_path + i)

        concat = pd.concat([train, supplemental])

        concat.to_csv(f"concatenated_data/{i}") 