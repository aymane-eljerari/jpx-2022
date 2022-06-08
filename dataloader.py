import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class JPXData(Dataset):
    def __init__(self, data_dir="modified_data", stock_list="tokenized_stock_list.csv", stock_prices="stock_prices.csv"):
        """
			data_dir        (string): Directory with where data is.
			stock_list      (string): File containing information about each stock.
            stock_prices    (string): File containing daily stock price data.
		"""
        self.data_dir           = data_dir
        self.stock_list         = pd.read_csv(data_dir + '/' + stock_list)
        self.stock_prices       = pd.read_csv(data_dir + '/' + stock_prices)

        self.stock_id = sorted(list(set(self.stock_prices["SecuritiesCode"])))

        # Sort dates chronologically and store them in self.__dates
        dates = []
        for i in list(set(self.stock_prices["Date"])):
            dates.append(datetime.strptime(i, '%Y-%m-%d').date())
        dates = sorted(dates)
    
        self.__dates            = dates
        self.__num_stocks       = len(self.stock_list)
        self.__indices          = {i: self.stock_id.index(i) for i in self.stock_id}
    
    def __len__(self):
        # Verify data set size
        return len(self.__dates) - 1
        
    def __getitem__(self, idx):
        # JPX Holiday Calendar 2017 [https://stock-market-holidays.org/2017-japan-exchange-group-holidays/]

        # Find the day corresponding to input idx 
        input_date  = self.__dates[idx]
        target_date = self.__dates[idx+1] 

        # Find index of first occurence of input_date in dataset
        input_date_first = input_date.strftime("%Y-%m-%d")

        # Find index of first occurence of target_date in dataset
        target_date_first = target_date.strftime("%Y-%m-%d")

        # Load Date specific data sorted by Securities Code
        input_data  = self.stock_prices.query("Date == @input_date_first").sort_values(by='SecuritiesCode')[["Open", 'SecuritiesCode']].reset_index()
        target_data = self.stock_prices.query("Date == @target_date_first").sort_values(by='SecuritiesCode')[["Open", 'SecuritiesCode']].reset_index()

        # Initialize tensors of zeros
        input_tensor  = torch.zeros(2000)
        target_tensor = torch.zeros(2000)

        # Write "Open" price data at it's given stock index
        for i in range(len(input_data)):
            input_index = self.__indices[input_data["SecuritiesCode"][i]]
            input_tensor[input_index] = input_data["Open"][i]
                
        for i in range(len(target_data)):
            target_index = self.__indices[target_data["SecuritiesCode"][i]]
            target_tensor[target_index] = target_data["Open"][i]
    
        # Convert nan to 0
        input_final  = input_tensor.nan_to_num(nan=0)
        target_final = target_tensor.nan_to_num(nan=0)

        # Normalize
        input_final  = (input_final - torch.mean(input_final)) / torch.std(input_final)
        target_final = (target_final - torch.mean(target_final)) / torch.std(target_final)

        return input_final, target_final
