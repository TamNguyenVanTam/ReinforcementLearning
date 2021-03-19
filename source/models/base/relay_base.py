"""
Relay Memmory for Holding Examples is implemented here 
Authors: TamNV
======================================================
"""

class MemoryBase(object):
	# Define Abstract Class for MemmoryBase
	def __init__(self, capability=100000):
		"""
		Initialize method for creating a new instance of memory based
		+ Params: capability: Integer
		"""
		self._capability = capability
		self._queue = {
						"s":[],
						"a":[],
						"ns":[],
						"r":[]
					}
		self._cur_size = 0
		self._pris = None

	def insert_samples(self, samples):
		"""
		Perform Inserting New Observations into Memory
		+ Params: samples
		"""
		pass

	def sel_samples(self, batch_size):
		"""
		Perform Selecting Observations from Previous
		"""
		pass

	def clr_queue(self):
		"""
		Perform Clearing Observations in this Queue
		"""
		pass
