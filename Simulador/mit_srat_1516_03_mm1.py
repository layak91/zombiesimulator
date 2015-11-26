#!/usr/bin/env python

from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt

class Simulator(object):
    """Simulation environment"""
    
    def __init__(self):
        self.now = 0
        self.queue = []
        self.processes = []
    
    def schedule(self, (delay, event)):
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

class Process(object):
    """Parent class"""
    
    def __init__(self, sim):
        self.sim = sim
        self.sim.processes.append(self)
    
    def run(self):
        raise NotImplementedError

class Monitor(object):
    """Statistics gathering"""
    
    def __init__(self, sim):
        self.sim = sim
        self.last = 0
        self.dt = []
        self.Qt = []
        self.Ut = []
    
    def observe(self, queue, server):
        self.dt.append(self.sim.now - self.last)
        self.last = self.sim.now
        self.Qt.append(queue)
        self.Ut.append(server)

class ExpGenerator(Process):
    """Generator with exponential arrivals"""
    
    def __init__(self, sim, mean_interarrival, out=None):
        super(ExpGenerator, self).__init__(sim)
        self.mi = mean_interarrival
        self.out = out
    
    def delay(self):
        return np.random.exponential(self.mi)
        
    def run(self):
        return [(self.delay(), self.next)]
    
    def next(self):
        return [(0, self.out.seize),
                (self.delay(), self.next)]

class ExpServer(Process):
    """Server with exponential service"""
    
    def __init__(self, sim, mean_service, mon=None):
        super(ExpServer, self).__init__(sim)
        self.ms = mean_service
        self.queue = 0
        self.busy = 0
        self.mon = mon
        
    def delay(self):
        return np.random.exponential(self.ms)
    
    def run(self):
        return []
    
    def seize(self):
        # Monitoring
        if self.mon:
            self.mon.observe(self.queue, self.busy)
        # Serve or enqueue
        if not self.busy:
            self.busy += 1
            return [(self.delay(), self.release)]
        else:
            self.queue += 1
            return []
    
    def release(self):
        # Monitoring
        if self.mon:
            self.mon.observe(self.queue, self.busy)
        # Serve another or halt
        if self.queue:
            self.queue -= 1
            return [(self.delay(), self.release)]
        else:
            self.busy -= 1
            return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple M/M/1')
    parser.add_argument('lambd',    type=float,                 help='interarrival rate')
    parser.add_argument('mu',       type=float,                 help='service rate')
    parser.add_argument('-s',       type=int,                   help='custom seed')
    parser.add_argument('-t',       type=int,   default=5000,   help='time to simulate')
    args = parser.parse_args()
    
    if args.s:
        np.random.seed(args.s)
    
    sim = Simulator()
    
    mon = Monitor(sim)
    server = ExpServer(sim, 1/args.mu, mon=mon)
    gen = ExpGenerator(sim, 1/args.lambd, out=server)
    
    sim.run(until=args.t)
    
    ### Figures
    dt = np.array(mon.dt)
    Ut = np.array(mon.Ut)
    Qt = np.array(mon.Qt)
    
    axis = plt.subplot()
    axis.set_title('M/M/1, $\lambda={}, \mu={}$'.format(args.lambd, args.mu))
    
    t = dt.cumsum()
    axis.step(t, Ut, label='Instantaneous server utilisation')
    axis.step(t, Qt, label='Instantaneous queue utilisation')
    N_average_t = ((Ut + Qt) * dt).cumsum() / t
    axis.plot(t, N_average_t, label='Average system utilisation')
    rho = (args.lambd)/(args.mu)
    axis.axhline(rho/(1-rho), linewidth=2, color='black', ls='--', label='Theoretical average')
    
    axis.set_xlabel('time')
    axis.set_ylabel('# of customers')
    axis.legend()
    
    plt.show()