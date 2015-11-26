#!/usr/bin/env python
	
from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Util.number import size

class population (object):
	"""Representation of the population"""
	def __init__(self, size, pInfec, pFriend):
		self.size = size
		self.pInfec = pInfec
		self.pFriend = pFriend
		self.matrix = self.generateMatrix()
		self.contagiados = self.generateContagiados()
		self.alive = np.array([1.0] * size)
		self.nalive = size
		
		
	def generateMatrix(self):
		matrix = np.zeros((self.size, self.size))
		for i in range(0, self.size):
			for j in range(i + 1, self.size):
				if np.random.random() < self.pFriend:
					matrix[i][j] = 1
					matrix[j][i] = 1
		print(matrix)
		return matrix
		
	def generateContagiados(self):
		contagiados = np.array([False] * self.size)
		for i in range(0, self.size):
			if np.random.random() < self.pInfec:
				contagiados[i]=True
				
		return contagiados

	
class Simulator (object):
	"""Simulation environment"""  
	def __init__(self, population):
		self.now = 0
		self.queue = []
		self.processes = []
		self.population = population
	
	def schedule(self, (delay, event, param)):
		heappush(self.queue, (self.now + delay, event, param))
		
	def run(self, ):
		# Initialisation
		if not self.processes:
			raise RuntimeError('no processes to run')
		for p in self.processes:
			self.schedule((0, p.run, None))
		
		# Loop
		while sum(self.population.contagiados)>0:
			aux=sum(self.population.contagiados)
			print(aux)
			print(self.population.contagiados)
			self.now, event, param = heappop(self.queue)
			# Monitoring here?
			if param != None:
				new_events = event(param)
			else:
				new_events = event()
			for ev in new_events:
				self.schedule(ev)

class Process (object):
	"""Parent class"""
	
	def __init__(self, sim):
		self.sim = sim
		self.sim.processes.append(self)
	
	def run(self):
		raise NotImplementedError

class InteractionsGeneretor (Process):
	"""Generator with exponential arrivals"""
	
	def __init__(self, sim, mean_interinterction, virus):
		super(InteractionsGeneretor, self).__init__(sim)
		self.mean = mean_interinterction
		self.virus = virus
	
	def delay(self):
		return np.random.exponential(self.mean)
	
	def run(self):
		return [(self.delay(), self.next, None)]
	
	def next(self):
		return [(0, self.virus.interaction, None),
			(self.delay(), self.next, None)]

class virus(Process):
	"""Server with exponential service"""
	
	def __init__(self, sim, population, meanDeath, mon=None):
		super(virus, self).__init__(sim)
		self.meanDeath = meanDeath
		self.mon = mon
		self.population = population

	def delay(self):
		return np.random.exponential(self.meanDeath)
	
	def infection(self):
		return (np.random.random() < self.population.pInfec)
	
	def run(self):
		return []
	
	def interaction(self):
		# Monitoring
		if self.mon:
			death = self.population.size-self.population.nalive
			self.mon.observe(self.population.nalive, death, sum(self.population.contagiados))
		# Random Interaction
		nodes = np.arange(0, self.population.size, 1)
		
		prob1 = self.population.alive / self.population.nalive
		aux = sum(prob1)
		#if aux == 1:
		pos1 = np.random.choice(nodes, p=prob1)
		#else:
		#	pos1 = 0
		if sum(self.population.matrix[pos1] * self.population.alive)>0:
			prob2 = (self.population.matrix[pos1] * self.population.alive) / sum(self.population.matrix[pos1] * self.population.alive)
			aux2 = sum(prob2)
			if (sum(prob2) == 1.0):
				pos2 = np.random.choice(nodes, p=prob2)
			else:
				pos2 = 0
		else:
			pos2 = 0
		if (self.population.alive[pos1] == 1.0) & (self.population.alive[pos2] == 1.0):
			if (self.population.contagiados[pos1]==True) & (self.population.contagiados[pos2] == False):
				if(self.infection()):
					self.population.contagiados[pos2] = True
					return[(self.delay(), self.death, pos2)]
				else:
					return[]
			if (self.population.contagiados[pos2] ==True) & (self.population.contagiados[pos1] == False):
				if(self.infection()):
					self.population.contagiados[pos1] = True
					return[(self.delay(), self.death, pos1)]
				else:
					return[]
			else:
				return[]
		else:
			return[]

	def death(self, pos):
		# Monitoring
		if self.mon:
			self.mon.observe(self.population.nalive, self.population.size-self.population.nalive, sum(self.population.contagiados))

		self.population.contagiados[pos] = False
		self.population.alive[pos] = 0
		self.population.nalive = self.population.nalive-1
		return[]
	
class Monitor(object):
	"""Statistics gathering"""

	def __init__(self, sim):
		self.sim = sim
		self.last = 0
		self.dt = []
		self.Alive = []
		self.Death = []
		self.Infected = []
		
	def observe(self, alive, death, infected):
		self.dt.append(self.sim.now - self.last)
		self.last = self.sim.now
		self.Alive.append(alive)
		self.Death.append(death)
		self.Infected.append(infected)
		
if __name__ == "__main__":
	import argparse
	
	pop = population(10, 0.3, 0.2)
	sim = Simulator(pop)
	mon = Monitor(sim)
	vir = virus(sim, pop, 0.5, mon)
	gen = InteractionsGeneretor(sim, 0.4, vir)
	
	sim.run()
	
	### Figures
	dt = np.array(mon.dt)
	alive = np.array(mon.Alive)
	death = np.array(mon.Death)
	infected = np.array(mon.Infected)
	
	axis = plt.subplot()
	#axis.set_title('M/M/1, $\lambda={}, \mu={}$'.format(args.lambd, args.mu))
	axis.set_title('Resultados')
	
	t = dt.cumsum()
	axis.step(t, alive, label='Number of alive people')
	axis.step(t, death, label='Number of death people')
	axis.step(t, infected, label='Number of infected people')
	
	axis.set_xlabel('time')
	axis.set_ylabel('# of people')
	axis.legend()
	
	plt.show()