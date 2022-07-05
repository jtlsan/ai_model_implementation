import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.hidden_size = hidden_size
		self.forget_gate_layer = nn.Linear(input_size + hidden_size, hidden_size)
		self.input_gate_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.temp_C_layer = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input_vector, hidden_state, cell_state):
        concat_vector = torch.concat([input_vector, hidden_state], axis=1)
        forget_gate = F.sigmoid(self.forget_gate_layer(concat_vector))
        input_gate = F.sigmoid(self.input_gate_layer(concat_vector))
        tmp_C = F.tanh(self.temp_C_layer(concat_vector))
        C = (cell_state*forget_gate) + (tmp_C*input_gate)
        
        output_gate = F.sigmoid(self.output_gate_layer(concat_vector))
        h = output_gate * F.tanh(C)
        
        return h, C
    
    def initHidden(self):
        return torch.zeros(1, hidden_size)
