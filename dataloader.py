import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class JPXData(Dataset):
    def __init__(self, batch_size, data_dir="modified_data", stock_list="tokenized_stock_list.csv", stock_prices="stock_prices.csv"):
        """
			data_dir        (string): Directory with where data is.
			stock_list      (string): File containing information about each stock.
            stock_prices    (string): File containing daily stock price data.
		"""
        self.data_dir           = data_dir
        self.stock_list         = pd.read_csv(data_dir + '/' + stock_list)
        self.stock_prices       = pd.read_csv(data_dir + '/' + stock_prices)
        self.batch_size         = batch_size
        
        # Sort dates chronologically and store them in self.__dates
        dates = []
        for i in list(set(self.stock_prices["Date"])):
            dates.append(datetime.strptime(i, '%Y-%m-%d').date())
        dates = sorted(dates)
    
        self.__dates            = dates
        self.__num_stocks       = len(self.stock_list)
    
    def __len__(self):
        # Verify data set size
        return len(self.stock_prices) // self.batch_size

    def __getitem__(self, idx):
        # JPX Holiday Calendar 2017 [https://stock-market-holidays.org/2017-japan-exchange-group-holidays/]

        # Find the day corresponding to input idx 
        input_date  = self.__dates[idx]
        target_date = self.__dates[idx+1]

        # Find index of first occurence of input_date in dataset
        input_date_first = list(self.stock_prices["Date"]).index(input_date.strftime("%Y-%m-%d"))

        # Find index of the last occurence of input_date in dataset
        reversed_data   = list(self.stock_prices["Date"])[::-1].index(input_date.strftime("%Y-%m-%d"))
        input_date_last = self.__len__() - 1 - reversed_data

        # Find index of first occurence of target_date in dataset
        target_date_first = list(self.stock_prices["Date"]).index(target_date.strftime("%Y-%m-%d"))

        # Find index of the last occurence of target_date in dataset
        reversed_data    = list(self.stock_prices["Date"])[::-1].index(target_date.strftime("%Y-%m-%d"))
        target_date_last = self.__len__() - 1 - reversed_data

        # print(f"IDX {idx} - Range of input  {input_date_first, input_date_last}, Number of Stocks on day {input_date}: {input_date_last - input_date_first + 1}")
        # print(f"IDX {idx} - Range of target {target_date_first, target_date_last}, Number of Stocks on day {target_date}: {target_date_last - target_date_first + 1}")

        input_data  = torch.tensor(self.stock_prices[["Open"]][input_date_first:input_date_last+1].values, dtype=torch.float).squeeze()
        target_data = torch.tensor(self.stock_prices[["Open"]][target_date_first:target_date_last+1].values, dtype=torch.float).squeeze()

        # Initialize dataframes of zeroes to append to input and target data
        input_zero_df  = torch.zeros(2000 - len(input_data), dtype=torch.float)
        target_zero_df = torch.zeros(2000 - len(target_data), dtype=torch.float)

        input_concat  = torch.cat((input_data, input_zero_df))
        target_concat = torch.cat((target_data, target_zero_df))

        # print(f"Input shape: {input_concat}")
        # print(f"Target shape: {target_concat}")


        # input_stacked  = torch.hstack((input_data[:, 0],input_data[:, 1]))
        # target_stacked = torch.hstack((target_data[:, 0], target_data[:, 1]))

        # print(f"Input shape: {input_stacked}")
        # print(f"Target shape: {target_stacked}")
        
        # # print(f"Input shape: {input_zero_df.shape}")
        # # print(f"Target shape: {target_zero_df.shape}")

        # input_concat  = pd.concat([input_data, input_zero_df], ignore_index=True)
        # target_concat = pd.concat([target_data, target_zero_df], ignore_index=True)

        # input_final  = input_concat.stack().values
        # target_final = target_concat.stack().values

        # print(f"Input shape: {input_final.shape}")
        # print(f"Target shape: {target_final.shape}")


        return input_concat, target_concat
