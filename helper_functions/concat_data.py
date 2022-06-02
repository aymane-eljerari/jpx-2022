import pandas as pd
import os

def concat_data(train_path = "data/train_files/", supplemental_path= "data/supplemental_files/"):
    """
    Concatenate Train and Supplemental
    Data into a signle file
    """

    path = "modified_data/"
    isdir = os.path.isdir(path)

    if not isdir:
        os.mkdir(path)
    
    files =  os.listdir(train_path)

    for i in files:

        train           = pd.read_csv(train_path + i)
        supplemental    = pd.read_csv(supplemental_path + i)

        concat = pd.concat([train, supplemental])

        concat.to_csv(f"modified_data/{i}") 
    
    print("Done concatenating all files!")