import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class JPXData(Dataset):
	def __init__(self, data, train=True, data_dir="modified_data", stock_list="tokenized_stock_list.csv", stock_prices="autoencoder_data.csv"):
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
			data = data[['Target', 'SecuritiesCode', 'Volume', 'AdjustedClose', 'AdjustedOpen',
       					'AdjustedHigh', 'AdjustedLow', 'BBANDS_upper', 'BBANDS_middle',
						'BBANDS_lower', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MIDPOINT',
						'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA', 'ADX', 'ADXR',
						'APO', 'AROON_down', 'AROON_up', 'AROONOSC', 'BOP', 'CCI', 'DX',
						'MACD_macd', 'MACD_macdsignal', 'MACD_macdhist', 'MFI', 'MINUS_DI',
						'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'RSI', 'STOCH_slowk',
						'STOCH_slowd', 'STOCHF_fastk', 'STOCHF_fastd', 'STOCHRSI_fastk',
						'STOCHRSI_fastd', 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV',
						'ATR', 'NATR', 'TRANGE', 'HT_DCPERIOD', 'HT_DCPHASE',
						'HT_PHASOR_inphase', 'HT_PHASOR_quadrature', 'HT_SINE_sine',
						'HT_SINE_leadsine', 'HT_TRENDMODE', 'BETA', 'CORREL', 'LINEARREG',
						'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV']].reset_index()


			input_data  = torch.tensor(data.iloc[:, 3:].values,dtype=torch.float)
			target_data = torch.tensor(data["Target"].values,dtype=torch.float)
			codes_data  = torch.tensor(data["SecuritiesCode"].values,dtype=torch.float)

			# get indecies for stocks present on date
			idx = torch.searchsorted(self.stock_id,codes_data)

			# standardize input
			means      = self.stock_means[idx]
			stds       = self.stock_stds[idx]
			input_data = torch.nan_to_num(input_data, nan=0)
			
			input_data = torch.divide(torch.subtract(input_data, means), stds)

			# remove nans from input
			# input_data = input_data.nan_to_num(nan=0)

			return input_data, target_data, idx

		def calc_io(date):

			input_data, target_data, idx = get_data(date)

			# tensor dimensions
			input_dim  = self.__num_stocks
			output_dim = self.__num_stocks

			# initialized tensors
			input_tensor  = torch.zeros(input_dim, 69)
			output_tensor = torch.empty(output_dim).fill_(float('NaN'))

			input_tensor[idx]  = input_data
			output_tensor[idx] = target_data

			return [input_tensor, output_tensor]

		data = np.array([calc_io(date) for date in self.dates])

		input_tensor  = data[:, 0]
		output_tensor = data[:, 1]

		input_tensor  = [np.hstack(i) for i in input_tensor]
		output_tensor = np.hstack(output_tensor)

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
			prices_raw = self.stock_prices.loc[self.stock_prices['SecuritiesCode'] == code, ['Volume', 'AdjustedClose', 'AdjustedOpen',
       					'AdjustedHigh', 'AdjustedLow', 'BBANDS_upper', 'BBANDS_middle',
						'BBANDS_lower', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MIDPOINT',
						'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA', 'ADX', 'ADXR',
						'APO', 'AROON_down', 'AROON_up', 'AROONOSC', 'BOP', 'CCI', 'DX',
						'MACD_macd', 'MACD_macdsignal', 'MACD_macdhist', 'MFI', 'MINUS_DI',
						'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'RSI', 'STOCH_slowk',
						'STOCH_slowd', 'STOCHF_fastk', 'STOCHF_fastd', 'STOCHRSI_fastk',
						'STOCHRSI_fastd', 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV',
						'ATR', 'NATR', 'TRANGE', 'HT_DCPERIOD', 'HT_DCPHASE',
						'HT_PHASOR_inphase', 'HT_PHASOR_quadrature', 'HT_SINE_sine',
						'HT_SINE_leadsine', 'HT_TRENDMODE', 'BETA', 'CORREL', 'LINEARREG',
						'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV']]

			# # removes NaNs
			# prices = [[j for j in i if not np.isnan(j)] for i in prices_raw]
			# # prices = prices_raw[np.logical_not(np.isnan(prices_raw))]
			# means, stds = [np.mean(i) for i in prices], [np.std(i) for i in prices]

			# return means and stds
			means = prices_raw.mean(axis=0, skipna=True).to_numpy()
			stds  = prices_raw.std(axis=0, skipna=True).to_numpy()
			
			return means, stds
		
		# calculates tensor [[mean_1,std_1], ... ,[mean_n,std_n]]
		means_stds = torch.tensor([np.array(calc_mean_std(code)) for code in self.stock_id.tolist()],dtype=torch.float)
		# [[mean_1,std_1], ... ,[mean_n,std_n]] -> ([[mean_1], ... ,[mean_n]] , [[std_1], ... ,[std_n]])
		means, stds = torch.tensor_split(means_stds, 2, dim=1)

		return means.squeeze(), stds.squeeze()

# if __name__ == '__main__':
# 	print('making dataset')
# 	data = JPXData_test()
# 	print('looping through the data')
# 	for i in range(len(data)):
# 		data[i]
