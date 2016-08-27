#python
'''
Run train command
'''

import sys, os
from parseLog import createBest
from helpers import *

if __name__=='__main__':	
	start = 0 
	to = 64
	pathToClusters = '/Users/ishay/VanillaCNN/clusters'

	if len(sys.argv)>1:
		pathToClusters = sys.argv[1]

	if len(sys.argv)>2:
		start = sys.argv[2]

	if len(sys.argv)>3:
		to = sys.argv[3]

	for i in range(start, to):
		trainCluster(i, pathToClusters)

