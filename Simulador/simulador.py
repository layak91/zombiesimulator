#!/usr/bin/env python

from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Util.number import size

class population (object):
	"""Representation of the population"""
	def __init__(self, size, pInfec):
		self.matrix=0
		self.size=size
		self.pInfec=pInfec
		self.generateMatrix()
		
		
	def generateMatrix(self):
		matrix = np.zeros((self.size,self.size))
		for i in range(0, self.size):
			for j in range(i+1, size):
				if np.random.random() < self.pInfec:
					matrix[i][j]=1
					matrix[j][i]=1
		return matrix

class Simulator(object):
    """Simulation environment"""
    
    def __init__(self):
        self.now = 0
        self.queue = []
        self.processes = []
    
    def schedule(self, delay, event):
        heappush(self.queue, (self.now + delay, event))
    
    def run(self, until=1000):
        # Initialisation
        if not self.processes:
            raise RuntimeError('no processes to run')
        for p in self.processes:
            self.schedule((0, p.run))
        
        # Loop
        while self.now < until:
            self.now, event = heappop(self.queue)
            # Monitoring here?
            new_events = event()
            for ev in new_events:
                self.schedule(ev)