import torch
import torch.nn as nn

class encoder(nn.Module):

	def __init__(self,layers=[4,3,2,1],activation=nn.SELU):
		super().__init__()

		linear_layers = [
			nn.Linear(*dims) for dims in zip(layers[:-1],layers[1:])
		]

		model = [
			j for i in linear_layers for j in [i,activation()]
		]

		self.net = nn.Sequential(*model)

	def forward(self,input):

		return self.net(input)

class decoder(nn.Module):
	def __init__(self,layers=[1,2,3,4],activation=nn.SELU):
		super().__init__()

		linear_layers = [
			nn.Linear(*dims) for dims in zip(layers[:-1],layers[1:])
		]

		model = [
			j for i in linear_layers[:-1] for j in [i,activation()]
		] + [linear_layers[-1]]

		self.net = nn.Sequential(*model)

	def forward(self,input):

		return self.net(input)

class autoencoder(nn.Module):
	def __init__(
			self,
			e_layers=[4,3,2,1],
			d_layers=[1,2,3,4]
		):

		super().__init__()

		if e_layers[0] != d_layers[-1]:
			Exception(f'the autoendoers input dimension ({e_layers[0]}) must match output dimension ({d_layers[-1]})')
		
		if e_layers[-1] != d_layers[0]:
			Exception(f'output dimension of encoder ({e_layers[-1]}) must match input dimension of decoder ({d_layers[0]})')
		
		self.encoder = encoder(layers=e_layers)
		self.decoder = decoder(layers=d_layers)
	
	def forward(self,inputs):
		latent = self.encoder(inputs)
		return self.decoder(latent)


# if __name__ == '__main__':
# 	from dataloader import JPXData
# 	data = JPXData()

# 	ae = autoencoder()

# 	for i in range(len(data)):
# 		input, _ = data[i]
# 		output = ae(input)