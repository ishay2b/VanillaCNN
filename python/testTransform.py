from numpy import *; _A=array
from matplotlib.pylab import *
ion()
from pickle import load, dump
import cv2
from  DataRow import *
from copy import copy

if __name__=='__main__':
	with open('testSetMini.pickle', 'r') as f:
		testSet = load(f)
	A1=testSet[1] ; l1 = A1.landmarks().reshape(-1,2)
	A2=testSet[7] ; l2 = A2.landmarks().reshape(-1,2)
	cropped = A1.transformedDataRow(l2)
	subplot(221),imshow(A1.show()),title('source')
	subplot(222),imshow(cropped.show()),title('tranformed and cropped')
	subplot(224),imshow(A2.show()),title('destination')
	show()

