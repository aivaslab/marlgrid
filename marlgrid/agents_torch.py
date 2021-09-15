import random
import numpy as np
from enum import IntEnum
from marlgrid.convlstm import ConvLSTM
import torch.nn as nn
import torch.nn.functional as F
import torch


class AC_LSTM():
	def __init__(self, siz, channels):
		self.siz = siz
		self.model = self.init_model(channels)
		self.model_target = self.init_model(channels)
		self.trainable = True

	def init_model(self, channels):
		#model.summary()
		return model

	def controllerAction(self, ob, preferences=None):
		#ob2 = [preferences[int(x)-1] if x > 0 else 0 for x in ob.flatten()]

		#ob should be all previous obs since lstm
		ob2 = self.model.predict(np.expand_dims(np.expand_dims(ob,0),0))
		#print(ob2)

		bestpos = np.argmax(ob2)
		#print('b',bestpos)
		return bestpos

class AC_ConvLSTM2D():
	def __init__(self, siz):
		self.siz = siz
		self.model = self.init_model()
		self.model_target = self.init_model()
		self.trainable = True

	def init_model(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
		return ConvLSTM(input_channels=32, 
						kernel_size=kernel_size,
						hidden_channels=hidden_channels, 
						step=step, 
						effective_step=effective_step)

	def controllerAction(self, ob, preferences=None):
		#ob2 = [preferences[int(x)-1] if x > 0 else 0 for x in ob.flatten()]

		#ob should be all previous obs since lstm
		ob2 = self.model.predict(np.expand_dims(np.expand_dims(ob,0),0))
		#print(ob2)

		bestpos = np.argmax(ob2)
		#print('b',bestpos)
		return bestpos

class ConvLSTMA3C(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size=3, step=1, effective_step=[1], outputs=2, window_size=5, siz=5):
		
		super(ConvLSTMA3C, self).__init__()
		self.recurrent = True
		self.steps = step
		self.siz = siz
		self.inputs = input_channels
		self.planes = hidden_channels[0]
		self.clstm = ConvLSTM(input_channels=input_channels, 
								hidden_channels=hidden_channels,
								kernel_size=kernel_size, 
								step=step,
								effective_step=effective_step, 
								window_size=5)
		self.pi1 = nn.Linear(self.planes * self.siz*  self.siz, 16)
		self.v1 = nn.Linear(self.planes * self.siz*  self.siz, 16)
		self.pi2 = nn.Linear(16, outputs)
		self.v2 = nn.Linear(16, 1)
		#self.softmax = nn.Softmax(dim=1)
		self.distribution = torch.distributions.Categorical
		self.initialized = False

	def forward(self, input):
		#print('forward', input.shape)
		x = input.reshape(-1, self.steps, self.inputs, self.siz, self.siz)
		#print(x.shape)
		x = self.clstm(x) 
		#print('clstm out', x.shape)
		x = x.reshape(-1, self.planes * self.siz*  self.siz)
		#print('asdf', x.shape)
		x = torch.relu(x)
		pi = torch.tanh(self.pi1(x))
		v = torch.tanh(self.v1(x))
		pi = self.pi2(pi)
		v = self.v2(v)
		#print(pi.shape, v.shape)
		return pi, v
		
	def choose_action(self, s):
		self.eval()
		#print(s.shape)
		logits, _ = self.forward(s)
		prob = F.softmax(logits, dim=1).data
		m = self.distribution(prob)
		return m.sample().numpy()[0]

	def loss_func(self, s, a, v_t):
		self.train()
		#print('ddd',s.shape)
		logits, values = self.forward(s)
		td = v_t - values
		c_loss = td.pow(2)
		
		probs = F.softmax(logits, dim=1)
		m = self.distribution(probs)
		exp_v = m.log_prob(a) * td.detach().squeeze()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss

class MentalNetA3C(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size=3, step=1, effective_step=[1], outputs=2, window_size=5, siz=5):
		
		super(MentalNetA3C, self).__init__()
		self.recurrent = True
		self.steps = step
		self.siz = siz
		self.inputs = input_channels
		self.planes = hidden_channels[0]
		self.resnet = rn.ResNet5(in_planes=input_channels, num_planes=self.planes)
		self.clstm = ConvLSTM(input_channels=self.planes, 
								hidden_channels=hidden_channels,
								kernel_size=3, 
								step=step,
								effective_step=effective_step, 
								window_size=5)
		self.fc1 = nn.Linear(self.planes * self.siz*  self.siz, outputs)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input):
		x = self.resnet(input.reshape(-1, self.inputs, self.siz, self.siz))
		x = self.clstm(x.reshape(-1, self.steps, self.planes, self.siz, self.siz)) #3 because avgpool? shouldn't it be 9?
		#print(x.shape)
		x = x.reshape(-1, self.planes * self.siz*  self.siz)
		x = torch.relu(x)
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.softmax(x)
		return x
		
	def choose_action(self, s):
		self.eval()
		#print(s.shape)
		logits, _ = self.forward(s)
		prob = F.softmax(logits, dim=1).data
		m = self.distribution(prob)
		return m.sample().numpy()[0]

	def loss_func(self, s, a, v_t):
		self.train()
		logits, values = self.forward(s)
		td = v_t - values
		c_loss = td.pow(2)
		
		probs = F.softmax(logits, dim=1)
		m = self.distribution(probs)
		exp_v = m.log_prob(a) * td.detach().squeeze()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss

'''class ConvLSTM2D(nn.Module):

	def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
		self.model = ConvLSTM(input_channels=input_channels, 
						kernel_size=kernel_size,
						hidden_channels=hidden_channels, 
						step=step, 
						effective_step=effective_step)

	def controllerAction(self, ob, preferences=None):
		#ob2 = [preferences[int(x)-1] if x > 0 else 0 for x in ob.flatten()]

		#ob should be all previous obs since lstm
		ob2 = self.model.predict(np.expand_dims(np.expand_dims(ob,0),0))
		#print(ob2)

		bestpos = np.argmax(ob2)
		#print('b',bestpos)
		return bestpos'''