"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
	if np_array.dtype != dtype:
		np_array = np_array.astype(dtype)
	return torch.from_numpy(np_array)

def set_init(layers):
	for layer in layers:
		nn.init.normal_(layer.weight, mean=0., std=0.1)
		nn.init.constant_(layer.bias, 0.)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
	if done:
		v_s_ = 0.			   # terminal
	else:
		v_s_ = lnet.forward(s_)[-1].data.numpy()[0, 0]

	buffer_v_target = []
	for r in br[::-1]:	# reverse buffer r
		v_s_ = r + gamma * v_s_
		buffer_v_target.append(v_s_)
	buffer_v_target.reverse()

	longinp = v_wrap(np.vstack(bs))
	#print(longinp.shape)
	loss = lnet.loss_func(
		longinp,
		v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else np.vstack(ba),
		#v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
		v_wrap(np.array(buffer_v_target)[:, None]))

	# calculate local gradients and push local parameters to global
	opt.zero_grad()
	loss.backward()

	for lp, gp in zip(lnet.parameters(), gnet.parameters()):
		gp._grad = lp.grad
	opt.step()

	lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name, printevery=1000):
	with global_ep.get_lock():
		global_ep.value += 1
		temp_ep = global_ep.value
	with global_ep_r.get_lock():
		if global_ep_r.value == 0.: #not moving average because of true thing
			global_ep_r.value = ep_r
		else:
			global_ep_r.value = global_ep_r.value * 0.995 + ep_r * 0.005
		temp_ep_r = global_ep_r.value
		res_queue.put(global_ep_r.value)
	if (temp_ep % printevery) == 0:
		print(
			name,
			"Ep:", temp_ep,
			"| Ep_r: %.3f" % temp_ep_r,
		)