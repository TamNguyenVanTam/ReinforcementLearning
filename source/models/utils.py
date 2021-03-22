"""
Untility functions are declared in This file
@Authors: TamNV
============================================
"""
import os
import json

def load_json_file(file):
	"""
	Load Json File
	+ Params: file : String
	+ Returns: content" Dictionary
	"""
	ext = file.split(".")[-1]
	if ext != "json":
		raise Exception("File isn't a Json File")
	if not os.path.exists(file):
		raise Exception("File doesn't exist")
	
	with open(file) as json_file:
		content = json.load(json_file)

	return content
