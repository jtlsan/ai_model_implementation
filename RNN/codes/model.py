import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.input_linear = nn.Linear(input_size, hidden_size)
		self.hidden_linear = nn.Linear(hidden_size, hidden_size)
		self.b = nn.Parameter(torch.randn(hidden_size))
		self.output_linear = nn.Linear(hidden_size, output_size)


	def forward(self, inputs, hidden):
		inputs_o = self.input_linear(inputs)
		hidden_o = F.tanh(inputs_o + self.hidden_linear(hidden) + self.b)
		output = self.output_linear(hidden_o)

		return output, hidden_o

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


