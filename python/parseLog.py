# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:19:50 2015

@author:Ishay Tubi

Parse log 
"""

import sys
import re
from matplotlib.pylab import show, title, plot

if len(sys.argv)>1:
    logFilePath = sys.argv[1]
else:
    logFilePath = '../log.txt'
    
floatReg='[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'

regex       =re.compile('.* Iteration (\d+), Testing net \(#\d+\)')
regexLos    =re.compile('.*Test net output #0: loss = (%s) (\* 1 = %s loss)'%(floatReg,floatReg))
regexLos    =re.compile('.*Test net output #0: loss = (%s)'%(floatReg,))

found=0

xAxis=[]
yAxis=[]

with open(logFilePath) as f:
    global result
    for line in f:
        xresult = regex.search(line)
        if (xresult):
            xAxis.append(xresult.groups(1)[0])
            
        yresult = regexLos.search(line)
        if (yresult):
            yAxis.append(yresult.groups(1)[0])

mnSize = min(len(xAxis),len(yAxis))
skip = 0

xAxis = xAxis[skip:mnSize]
yAxis = yAxis[skip:mnSize]

plot(xAxis, yAxis)
title('test loss over iterations')
show()

import mpldatacursor
mpldatacursor.datacursor()


