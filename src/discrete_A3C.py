"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from src.utils_a3c import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from src.shared_adam import SharedAdam
import time
import numpy as np
#import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"
import tqdm


class CNet(nn.Module):
	def __init__(self, s_dim, a_dim, hidden=6):
		super(CNet, self).__init__()
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.c1 = nn.Conv2d(3, 16, 3, stride=2)
		self.c2 = nn.Conv2d(16, 8, 3, stride=2)
		self.pi1 = nn.Linear(s_dim, hidden)
		self.pi2 = nn.Linear(hidden, a_dim)
		self.v1 = nn.Linear(s_dim, hidden)
		self.v2 = nn.Linear(hidden, 1)
		set_init([self.pi1, self.pi2, self.v1, self.v2])
		self.distribution = torch.distributions.Categorical
		self.initialized = True

	def forward(self, x):
		#print(x.shape)
		x = x.permute(0,1,4,2,3)
		x = torch.squeeze(x, dim=1)
		x = self.c1(x)
		x = self.c2(x)
		#print(x.shape)
		x = torch.flatten(x, start_dim=1)
		#print(x.shape)
		pi1 = torch.tanh(self.pi1(x))
		logits = self.pi2(pi1)
		v1 = torch.tanh(self.v1(x))
		values = self.v2(v1)
		return logits, values

	def choose_action(self, s):
		self.eval()
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

class Net(nn.Module):
	def __init__(self, s_dim, a_dim, hidden=16):
		super(Net, self).__init__()
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.pi1 = nn.Linear(s_dim, hidden)
		self.pi2 = nn.Linear(hidden, a_dim)
		self.v1 = nn.Linear(s_dim, hidden)
		self.v2 = nn.Linear(hidden, 1)
		set_init([self.pi1, self.pi2, self.v1, self.v2])
		self.distribution = torch.distributions.Categorical
		self.initialized = True

	def forward(self, x):
		#print(x.shape)
		x = torch.flatten(x, start_dim=1)
		#print(x.shape)
		pi1 = torch.tanh(self.pi1(x))
		logits = self.pi2(pi1)
		v1 = torch.tanh(self.v1(x))
		values = self.v2(v1)
		return logits, values

	def choose_action(self, s):
		self.eval()
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


class Worker(mp.Process):
	def __init__(self, 
		gnet, 
		opt, 
		global_ep, 
		global_ep_r, 
		res_queue, 
		name,
		env, 
		episodes, 
		gamma=0.9, 
		update_iter=100, 
		modelClass=None, 
		margs=[None], 
		device=None, 
		steps=8):
		super(Worker, self).__init__()
		self.name = 'w%02i' % name
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
		self.gnet, self.opt = gnet, opt
		self.lnet = modelClass(*margs, )		   # local network, not on cuda?
		self.env = env#gym.make('CartPole-v0').unwrapped
		self.episodes = episodes
		self.update_iter = update_iter
		self.gamma = gamma
		self.steps = steps
		if not self.gnet.initialized:
			#print('initializing gnet')
			s = self.env.reset()
			buffer_s = []
			for x in range(self.steps):
				buffer_s.append(np.zeros(s[0][np.newaxis, :].shape))
			s = torch.FloatTensor([buffer_s])
			self.gnet.choose_action(s)
			self.gnet.initialized = True

	def run(self):
		total_step = 1
		while self.g_ep.value < self.episodes:
			s = self.env.reset()

			buffer_bigs, buffer_s, buffer_a, buffer_r = [], [], [], []
			ep_r = 0.

			buffer_s = [np.zeros(s[0][np.newaxis, :].shape) for _ in range(self.steps)]
			while True:
				if True: # and self.g_ep.value == self.episodes-2:
					print('rend')
					print(self.env)
					self.env.render()
					time.sleep(1)

				buffer_s.append(s[0])
				s2 = torch.FloatTensor([buffer_s[-self.steps:]])
				buffer_bigs.append(s2)

				#print(self.name, 'running...', total_step, s2.shape)
				a = [self.lnet.choose_action(s2),]
				#a = self.lnet.choose_action(v_wrap(s[None, :]))

				s_, r, done, _ = self.env.step(a)

				ep_r += r
				buffer_a.append(a[0])
				buffer_r.append(r)
				if ((total_step % self.update_iter) == 0) or done:  # update global and assign to local net
					# sync

					s_2 = torch.FloatTensor([buffer_s[-self.steps:]])

					push_and_pull(self.opt, self.lnet, self.gnet, done, s_2, buffer_bigs, buffer_a, buffer_r, self.gamma)
					buffer_bigs, buffer_s, buffer_a, buffer_r = [], [], [], []

					buffer_s = [np.zeros(s[0].shape) for _ in range(self.steps)]

					if done:  # done and print information
						record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, printevery=1e4)
						break

				s = s_
				total_step += 1
		self.res_queue.put(None)

def train_agent(name, 
		env, 
		model, 
		episodes, 
		gamma=0.9, 
		modelClass=None, 
		margs=[None], 
		device=None, 
		steps=8, 
		max_workers=16, 
		num_checkpoints=10, 
		path=''
		):
	gnet = model		# global network
	gnet.share_memory()		 # share the global parameters in multiprocessing

	opt = SharedAdam(gnet.parameters(), lr=1e-5, betas=(0.92, 0.999))	  # global optimizer
	global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
	update_iter = 101
	try:
		mp.set_start_method('spawn', force=True)
	except RuntimeError:
		pass

	# parallel training
	# mp.set_start_method('spawn')
	print('train agent start')
	num = min(mp.cpu_count(), max_workers)

	workers = [Worker(gnet, 
					opt, 
					global_ep, 
					global_ep_r, 
					res_queue, 
					i, 
					env, 
					episodes, 
					gamma, 
					update_iter, 
					modelClass, 
					margs, 
					device, 
					steps
					) for i in range(num)]
	print('beginning training for', episodes, 'episodes length', steps, 'with', num, 'workers')
	[w.start() for w in workers]
	res = []					# record episode reward to plot


	tq = tqdm.tqdm(total=episodes)
	checkpoint_every = episodes / num_checkpoints
	p = 1
	while True:
		#print(len(res_queue))
		r = res_queue.get()
		#print(r)
		if r is not None:
			res.append(r)
			tq.update(1)
			if p % checkpoint_every == 0:
				num = int(p/checkpoint_every)
				print('saving checkpoint', num)
				torch.save(gnet.state_dict(), os.path.join(path, 'a_chkpt_'+str(num)+'.pth'))
			p += 1
		else:
			break
	tq.close()
	[w.join() for w in workers]
	print('done')
	#print('saving weights')
	#torch.save(gnet.state_dict(), name + '_weights.pth')

	return res