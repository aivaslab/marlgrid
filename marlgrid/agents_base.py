import random
import numpy as np
from enum import IntEnum

class Directions(IntEnum):
	NW = 0
	N = 1
	NE = 2
	W = 3
	X = 4
	E = 5
	SW = 6
	S = 7
	SE = 8

class ADirections(IntEnum):
	E = 0
	N = 1
	W = 2
	S = 3
	X = 4

class Agent():
	def __init__(self, controller):
		self.controller = controller
		self.viewsize = controller.viewsize
		if self.viewsize == 5:
			viewport = 2
		elif self.viewsize == 3:
			viewport = 1
		elif self.viewsize == 7:
			viewport = 3
		self.preferences = [x + 1 for x in range(3)] #randomly decide preferences #TODO: make number a parameter

		self.direction = 0
		self.pos = (0,0)
		self.viewport = 1 #number of tiles to either side


	def action(self, ob):
		return self.controller.controllerAction(ob) #preferences here?

	def reset_prefs(self):
		random.shuffle(self.preferences)

	def hardcode_prefs(self):
		self.preferences.sort()

class AC():
	def __init__(self, viewsize):
		self.viewsize = viewsize
		self.trainable = False
		self.recurrent = False

	def controllerAction(self, ob, preferences=None):
		'''string = ''
		for y in range(ob.shape[1]):
			for x in range(ob.shape[0]):
				string += str(int(ob[x,y]))
			string += '\n'
		print(string)''
		print(ob.transpose().flatten())'''
		assert preferences is not None
		ob2 = [preferences[int(x)-1] if x > 0 else 0 for x in ob.transpose().flatten()]
		#print(ob2)
		bestscore = np.max(ob2)
		bestpos = np.argmax(ob2)
		ret = 0
		if bestscore == 0:
			ret = random.randint(0,3)
		elif bestpos == Directions.NW:
			ret = random.choice([1,2])
		elif bestpos == Directions.N:
			ret = 1
		elif bestpos == Directions.NE:
			ret = random.choice([0,1])
		elif bestpos == Directions.W:
			ret = 2
		elif bestpos == Directions.X:
			ret = 4
		elif bestpos == Directions.E:
			ret = 0
		elif bestpos == Directions.SW:
			ret =  random.choice([2,3])
		elif bestpos == Directions.S:
			ret = 3
		else:
			ret = random.choice([3,0])
		#print(bestpos, ret)
		return ret

class AC_Deterministic():
	def __init__(self, viewsize):
		self.trainable = False
		self.viewsize = viewsize
		self.recurrent = False

	def controllerAction(self, ob, preferences, maxindex=3): #todo: 4 = numagents+numobs

		null = np.zeros((self.viewsize,self.viewsize,1))
		
		ob_zeros = np.c_[null, ob[...,:maxindex]]
		obmax = np.argmax(np.arange(1,maxindex+2)*ob_zeros[...,:maxindex+1], axis=2)
		ob2 = [preferences[int(x)-1] if (x > 0) else -1 for x in obmax.transpose().flatten()]

		bestscore = np.max(ob2)
		bestpos = np.argmax(ob2)
		ret = 0
		if bestscore == -1:
			ret = ADirections.E
		elif bestpos == Directions.NW or bestpos == Directions.N:
			ret = ADirections.N
		elif bestpos == Directions.NE or bestpos == Directions.E:
			ret = ADirections.E
		elif bestpos == Directions.W or bestpos == Directions.SW:
			ret = ADirections.W
		elif bestpos == Directions.X:
			ret = ADirections.X
		elif bestpos == Directions.S or bestpos == Directions.SE:
			ret = ADirections.S
		#print(bestpos, ret)
		return ret

class AC_Network():
	def __init__(self, modelClass, modelParams, view_size, device):
		self.model = modelClass(*modelParams, )
		self.trainable = True
		self.viewsize = view_size
		self.recurrent = True

	def controllerAction(self, ob):
		#return self.model.predict(ob)
		return self.model.choose_action(ob.cuda())

	def set_weights(self, weights):
		self.model.load_state_dict(weights)