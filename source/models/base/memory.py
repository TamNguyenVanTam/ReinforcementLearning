"""
Memory Class is declared here
Authors: TamNV
=============================
"""
import random
from base.relay_base import MemoryBase

class Memory(MemoryBase):
	"""
	Declare Memory for Training Deep Reinforcement Learning
	"""
	def __init__(self, capability):
		super(Memory, self).__init__(capability)

	
	def insert_samples(self, samples):
		"""
		Insert Samples to Queue
		+ Params: samples: Dictionary
		+ Returns: None 
		"""
		if self._queue.keys() != samples.keys():
			raise Exception("Inserted Samples are't the same format")

		num_sams = len(samples["s"])

		for key in self._queue.keys():
			self._queue[key] = samples[key] + self._queue[key]
		
		if self._cur_size + num_sams <= self._capability:
			self._cur_size += num_sams
			return 

		# Remove Over Samples in a Queue
		for key in self._queue.keys():
			self._queue[key] = self._queue[key][0:self._capability]
		
		self._cur_size = self._capability
		return

	def sel_samples(self, batch_size):
		"""
		Select n samples from queue
		+ Params: batch_size: Integer
		"""
		batch = {"s":[], "a":[], "ns":[], "r":[], "d":[]}
		
		if batch.keys() != self._queue.keys():
			raise Exception("Format of Batch and Queue must be same")

		if self._cur_size <= batch_size:
			return self._queue 

		idxs = random.sample(list(range(self._cur_size)), batch_size)

		for idx in idxs:
			for key in self._queue.keys():
				batch[key].append(self._queue[key][idx])
		
		return batch

	def clr_queue(self):
		"""
		Perform Clearing Observations
		"""
		self._cur_size = 0
		for key in self._queue.keys():
			self._queue[key] = []
