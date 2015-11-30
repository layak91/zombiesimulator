#!/usr/bin/env python
	
from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Util.number import size
from duplicity.tempdir import default

class population (object):
	"""Representation of the population"""
	def __init__(self, size, pInfec, pFriend, pInfecIni):
		self.size = size
		self.pInfec = pInfec
		self.pFriend = pFriend
		self.pInfecIni = pInfecIni
		self.matrix = self.generateMatrix()
		self.contagiados = self.generateContagiados()
		self.alive = np.array([1.0] * size)
		self.nalive = size
		self.cured = np.array([0] * self.size)
		
		
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
			if np.random.random() < self.pInfecIni:
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
		
	def run(self):
		# Initialisation
		if not self.processes:
			raise RuntimeError('no processes to run')
		for p in self.processes:
			self.schedule((0, p.run, None))
		
		# Loop
		while sum(self.population.contagiados)>0:
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
		contador=0
		for i in self.sim.population.contagiados:
			if i==True:
				heappush(self.sim.queue, (self.sim.now + self.virus.delay(), self.virus.death, contador))
			contador=contador+1
		
	def delay(self):
		return np.random.exponential(self.mean)
	
	def run(self):
		return [(self.delay(), self.next, None)]
	
	def next(self):
		return [(0, self.virus.interaction, None),
			(self.delay(), self.next, None)]

class virus(Process):
	"""Virus actions"""
	
	def __init__(self, sim, population, meanDeath, goberment, mon=None):
		super(virus, self).__init__(sim)
		self.meanDeath = meanDeath
		self.mon = mon
		self.population = population
		self.goberment = goberment

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
			self.mon.observe(self.population.nalive, death, sum(self.population.contagiados), self.goberment.defcon, self.goberment.cure)
		# Random Interaction
		nodes = np.arange(0, self.population.size, 1)
		
		prob1 = self.population.alive / self.population.nalive
		if sum(prob1) >= 0.99999999999999:
			pos1 = np.random.choice(nodes, p=prob1)
		else:
			pos1 = 0
			print("EROR: Probabilidad 1 incorrecta")
		if sum(self.population.matrix[pos1] * self.population.alive)>0:
			prob2 = (self.population.matrix[pos1] * self.population.alive) / sum(self.population.matrix[pos1] * self.population.alive)
			if (sum(prob2) >= 0.99999999999999):
				pos2 = np.random.choice(nodes, p=prob2)
			else:
				print("EROR: Probabilidad 2 incorrecta")
				pos2 = 0
		else:
			pos2 = 0
		if (self.population.alive[pos1] == 1.0) & (self.population.alive[pos2] == 1.0):
			if (self.population.cured[pos1]==0) & (self.population.cured[pos2]==0):
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
		else:
			return[]

	def death(self, pos):
		# Monitoring
		if self.mon:
			death = self.population.size-self.population.nalive
			self.mon.observe(self.population.nalive, death, sum(self.population.contagiados), self.goberment.defcon, self.goberment.cure)

		self.population.contagiados[pos] = False
		self.population.alive[pos] = 0
		self.population.nalive = self.population.nalive-1
		return[(0, self.goberment.newDeath, None)]
	
class Monitor(object):
	"""Statistics gathering"""

	def __init__(self, sim):
		self.sim = sim
		self.last = 0
		self.dt = []
		self.Alive = []
		self.Death = []
		self.Infected = []
		self.defcon = []
		self.cure = []
		
	def observe(self, alive, death, infected, defcon, cure):
		self.dt.append(self.sim.now - self.last)
		self.last = self.sim.now
		self.Alive.append(alive)
		self.Death.append(death)
		self.Infected.append(infected)
		self.defcon.append(defcon)
		self.cure.append(cure)
		
class Goberment(object):
	"""Goberment actions"""
	
	def __init__(self, sim, population, mon=None):
		#super(Goberment, self).__init__(sim)
		self.population = population
		self.defcon = 0.0
		self.investigating = False
		self.cure = 0.0
		self.mon = mon
		self.sim = sim
		
	def newDeath (self):
		if self.mon:
			death = self.population.size-self.population.nalive
			self.mon.observe(self.population.nalive, death, sum(self.population.contagiados), self.defcon, self.cure)
			
		if self.cure<100.0:
			
			if(self.defcon < 100):
				#self.defcon=self.defcon + float(death)/float(self.population.size)
				self.defcon=self.defcon + 1
				
			if self.investigating == False:
				if np.random.random() < self.defcon/100:
					self.investigating = True
				return []
			else:
				self.cure=self.cure + self.defcon*self.population.size/10/100
				if self.cure > 100.0:
					self.cure = 100.0
					
				if self.cure<100:
					return[]
				else:
					return[(0, self.deliverCure, None)]
		else:
			return[]
				
	def deliverCure (self):
		if self.mon:
			death = self.population.size-self.population.nalive
			self.mon.observe(self.population.nalive, death, sum(self.population.contagiados), self.defcon, self.cure)

		prob = (self.population.alive * np.invert(self.population.cured)) / sum(self.population.alive * np.invert(self.population.cured))
		nodes = np.arange(0, self.population.size, 1)
		pos = np.random.choice(nodes, p=prob)
		self.population.contagiados[pos]=False
		self.population.cured[pos]=1.0
		return[(0.1, self.deliverCure, None)]
	
	
if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='Simple M/M/1')
	parser.add_argument('-pFriend',	   			type=float,	default=0.2,					help='Probabilidad de amigos')
	parser.add_argument('-pInfec',	    		type=float,	default=0.3,					help='Probabilidad de contagio')
	parser.add_argument('-pInfecIni',	    		type=float,	default=0.2,					help='Probabilidad de infeccion inicial')
	parser.add_argument('-meanDeath',			type=float,	default=60,		    		help='Ratio Muertes')
	parser.add_argument('-mean_interinterction',	type=float,	default=7,		    		help='Ratio interacciones')
	parser.add_argument('-size',         		type=int,	default=100,						help='Tamano matriz')
	args = parser.parse_args()
	
	pop = population(args.size, args.pInfec, args.pFriend, args.pInfecIni)
	sim = Simulator(pop)
	mon = Monitor(sim)
	gob = Goberment(sim, pop, mon)
	vir = virus(sim, pop, args.meanDeath, gob, mon)
	#aux = float(args.mean_interinterction/float(pop.size))
	gen = InteractionsGeneretor(sim, args.mean_interinterction/float(pop.size), vir)
	
	sim.run()
	
	### Figures
	dt = np.array(mon.dt)
	alive = np.array(mon.Alive)
	death = np.array(mon.Death)
	infected = np.array(mon.Infected)
	defcon = np.array(mon.defcon)
	cure = np.array(mon.cure)
	
	axis = plt.subplot()
	#axis.set_title('M/M/1, $\lambda={}, \mu={}$'.format(args.lambd, args.mu))
	axis.set_title('Resultados')
	
	t = dt.cumsum()
	axis.step(t, alive, label='Number of alive people')
	axis.step(t, death, label='Number of death people')
	axis.step(t, infected, label='Number of infected people')
	axis.step(t, defcon, label='Level of Defcon')
	axis.step(t, cure, label='Level of Cure')
	
	axis.set_xlabel('Days')
	axis.set_ylabel('# of people')
	axis.legend()
	
	plt.show()