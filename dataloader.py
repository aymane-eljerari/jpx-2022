import os
import pandas as pd
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
        
        # Sort dates chronologically and store them in self.__dates
        dates = []
        for i in list(set(self.stock_prices["Date"])):
            dates.append(datetime.strptime(i, '%Y-%m-%d').date())
        dates = sorted(dates)
    
        self.__dates              = dates
    
    def __len__(self):
        # Verify data set size
        return len(self.stock_prices)

    def __getitem__(self, idx):
        # JPX Holiday Calendar 2017 [https://stock-market-holidays.org/2017-japan-exchange-group-holidays/]

        # Find the day corresponding to input idx 
        input_date = self.__dates[idx]
        target_date = self.__dates[idx+1]

        # Find index of first occurence of input_date in dataset
        input_date_first = list(self.stock_prices["Date"]).index(input_date.strftime("%Y-%m-%d"))

        # Find index of the last occurence of input_date in dataset
        reversed_data = list(self.stock_prices["Date"])[::-1].index(input_date.strftime("%Y-%m-%d"))
        input_date_last = self.__len__() - 1 - reversed_data

        # Find index of first occurence of target_date in dataset
        target_date_first = list(self.stock_prices["Date"]).index(target_date.strftime("%Y-%m-%d"))

        # Find index of the last occurence of target_date in dataset
        reversed_data = list(self.stock_prices["Date"])[::-1].index(target_date.strftime("%Y-%m-%d"))
        target_date_last = self.__len__() - 1 - reversed_data

        print(f"Range of input  {input_date_first, input_date_last}, Number of Stocks on day {input_date}: {input_date_last - input_date_first + 1}")
        print(f"Range of target {target_date_first, target_date_last}, Number of Stocks on day {target_date}: {target_date_last - target_date_first + 1}")
        
        input_data = self.stock_prices[["SecuritiesCode", "Open"]][input_date_first:input_date_last+1]
        target_data = self.stock_prices[["SecuritiesCode", "Open"]][target_date_first:target_date_last+1]

        return input_data, target_data

