
import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size, cuda=False):
		super(ConvLSTMCell, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.device = 'cpu'
		#self.num_features = 12 #unused variable?

		self.padding = int((kernel_size - 1) / 2)

		self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
		self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
		self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
		self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
		self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
		self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
		self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
		self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

		self.Wci = None
		self.Wcf = None
		self.Wco = None


	def forward(self, x, h, c):
		ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
		cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
		cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
		co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
		ch = co * torch.tanh(cc)
		return ch, cc

	def init_hidden(self, batch_size, hidden, shape):
		if self.Wci is None:
			self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], )).to(self.device)
			self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], )).to(self.device)
			self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], )).to(self.device)
		else:
			assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
			assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
		return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], )).to(self.device),#device=torch.device("cuda"))),
				Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], )).to(self.device))#device=torch.device("cuda"))))


class ConvLSTM(nn.Module):
	# input_channels corresponds to the first input feature map
	# hidden state is a list of succeeding lstm layers.
	def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], window_size=5, device='cpu'):
		super(ConvLSTM, self).__init__()
		self.input_channels = [input_channels] + hidden_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.num_layers = len(hidden_channels)
		self.step = step
		self.effective_step = effective_step
		self._all_layers = []
		for i in range(self.num_layers):
			name = 'cell{}'.format(i)
			cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
			cell.device = device
			setattr(self, name, cell)
			self._all_layers.append(cell)

		self.device = device


	def forward(self, input):
		internal_state = []
		input = input.to(self.device)
		for step in range(self.step):
			#x = input #original
			x = input[:,step]
			for i in range(self.num_layers):
				# all cells are initialized in the first step
				name = 'cell{}'.format(i)
				if step == 0:
					#print(x.size())
					bsize, _, height, width = x.size() #original, for 4-dimensional input 
					#bsize, frames, channels, height, width = x.size() 
					(h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
															 shape=(height, width))
					internal_state.append((h, c))

				# do forward
				(h, c) = internal_state[i]
				x, new_c = getattr(self, name)(x, h, c)
				internal_state[i] = (x, new_c)
			# only record effective steps
		#print(x.shape)
		return x

		#return outputs, (x, new_c)
'''
convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
					effective_step=[4]).cuda()
loss_fn = torch.nn.MSELoss()

input = Variable(torch.randn(1, 512, 64, 32)).cuda()
target = Variable(torch.randn(1, 32, 64, 32)).double().cuda()

output = convlstm(input)
output = output[0][0].double()
res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
print(res)'''