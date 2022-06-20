import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class JPXData(Dataset):
	def __init__(self, data, train=True, data_dir="modified_data", stock_list="tokenized_stock_list.csv", stock_prices="stock_prices.csv"):
		"""
			data_dir        (string): Directory with where data is.
			stock_list      (string): File containing information about each stock.
			stock_prices    (string): File containing daily stock price data.
		"""
		self.data_dir           = data_dir
		self.stock_list         = pd.read_csv(data_dir + '/' + stock_list)
		self.stock_prices       = data#pd.read_csv(data_dir + '/' + stock_prices)
		self.train				= train
		self.stock_id           = sorted(list(set(self.stock_prices["SecuritiesCode"])))
		self.indices            = {i: self.stock_id.index(i) for i in self.stock_id}
		self.stock_id           = torch.tensor(self.stock_id)
		self.dates              = self.__process_dates()
		self.__num_stocks       = len(self.stock_id)

		# Store mean and variance for each stock to normalize
		self.stock_means, self.stock_stds  = self.__calc_means_stds()

		self.inputs , self.targets = self.__process_data()

		
	def __len__(self):
		# Verify data set size
		return len(self.dates)
	
	def __getitem__(self, idx):
		if self.train:
			return self.inputs[idx], self.targets[idx], [self.dates[idx].strftime('%Y-%m-%d')]
		else:
			return self.inputs[idx], self.targets[idx], [self.dates[idx].strftime('%Y-%m-%d')]
	
	def __process_data(self):
		def get_data(date):
			input_date = date.strftime("%Y-%m-%d")

			# extract input data
			data = self.stock_prices.query("Date == @input_date")
			data = data.sort_values(by='SecuritiesCode')
			data = data[["AdjustedClose", "Target", 'SecuritiesCode']].reset_index()

			input_data  = torch.tensor(data[ "AdjustedClose"].values,dtype=torch.float)
			target_data = torch.tensor(data["Target"].values,dtype=torch.float)
			codes_data  = torch.tensor(data["SecuritiesCode"].values,dtype=torch.float)

			# get indecies for stocks present on date
			idx = torch.searchsorted(self.stock_id,codes_data)

			# standardize input
			means      = self.stock_means[idx]
			stds       = self.stock_stds[idx]
			input_data = (input_data - means) / stds

			# remove nans from input
			input_data = input_data.nan_to_num(nan=0)

			return input_data, target_data, idx

		def calc_io(date):
			# tensor dimensions
			input_dim  = self.__num_stocks
			output_dim = self.__num_stocks

			# initialized tensors
			input_tensor  = torch.zeros(input_dim)
			output_tensor = torch.empty(output_dim).fill_(float('NaN'))

			input_data, target_data, idx = get_data(date)

			input_tensor[ idx] = input_data
			output_tensor[idx] = target_data

			return [input_tensor, output_tensor]

		input_tensor, output_tensor = zip(*[calc_io(date) for date in self.dates])

		input_tensor  = torch.vstack(input_tensor)
		output_tensor = torch.vstack(output_tensor)

		return input_tensor, output_tensor


	def __process_dates(self):
		dates = []
		for i in list(set(self.stock_prices["Date"])):
			dates.append(datetime.strptime(i, '%Y-%m-%d').date())
		return sorted(dates)
	
	def __calc_means_stds(self):

		# calculates mean given stock code
		def calc_mean_std(code):
			# gets prices with NaNs
			prices_raw = self.stock_prices.loc[self.stock_prices['SecuritiesCode'] == code]["AdjustedClose"].to_numpy()

			# removes NaNs
			prices = prices_raw[np.logical_not(np.isnan(prices_raw))]

			return np.mean(prices), np.std(prices)
		
		# calculates tensor [[mean_1,std_1], ... ,[mean_n,std_n]]
		means_stds = torch.tensor([list(calc_mean_std(code)) for code in self.stock_id.tolist()],dtype=torch.float)

		# [[mean_1,std_1], ... ,[mean_n,std_n]] -> ([[mean_1], ... ,[mean_n]] , [[std_1], ... ,[std_n]])
		means, stds = torch.tensor_split(means_stds, 2, dim=1)

		return means.squeeze(), stds.squeeze()

if __name__ == '__main__':
	print('making dataset')
	data = JPXData_test()
	print('looping through the data')
	for i in range(len(data)):
		data[i]
