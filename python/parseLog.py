# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:19:50 2015

@author:Ishay Tubi

Parse log, Create best snap shot from log 
"""

import sys
import re
import numpy as np
from helpers import *

if __name__=='__main__':
    from matplotlib.pylab import show, title, plot

    if len(sys.argv)>1:
        clusterPath = sys.argv[1]
    else:
        clusterPath = '/Users/ishay/VanillaCNN/clusters/0'

    xAxis, yAxis = parseLog(os.path.join(clusterPath,'clusterLog.txt'))
    minIndex = np.argmin(yAxis)
    x, y = xAxis[minIndex], xAxis[minIndex]
    print "min error:(x,y):", x,y
    import mpldatacursor
    mpldatacursor.datacursor()
    plot(xAxis, yAxis)
    title('test loss over iterations')
    show()

    xAxis, yAxis = createBest(clusterPath) 


