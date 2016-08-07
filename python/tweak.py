import timeit
import numpy as np
_A = np.array  # A shortcut to creating arrays in command line 
import os
import cv2
import sys
from pickle import load, dump

#Import helper functions
from helpers import *
from DataRow import *
from pdb import set_trace

#from matplotlib.pylab import *
FULL_STEPS=[
'createTrainingSetPickle', 
'createGMM', 
'createTrainClusters',
'createTestSetPickle',
'createTestClusters',
'trainCLusters',
'runTweakTest']

STEPS=['', 'viewCluster']
#STEPS=['createTrainingSetPickle', 'createTrainClusters','createTestClusters','runTweakTest']

start = timeit.default_timer() # total running time 

# Create the MTFL benchmark
if 'createTrainingSetPickle' in STEPS:  
    createTrainingSetPickle()

if 'createTestSetPickle' in STEPS:
    print "Creating test set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d rows from test" % len(dataRowsTest_CSV)
    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print "Original test:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    with open('testSetPickle.pickle','w') as f:
        dump(dataRowsTestValid, f)

if 'createGMM' in STEPS:
    createGMM()

#Load GMM
with open('gmm.pickle') as f:
    gmm=load(f)

if 'createTrainClusters' in STEPS:
    print "Creating train set"
    with open('trainSetMTFL.pickle','r') as f:
        dataRows = load(f)
    createClusteredData(dataRows=dataRows, 
        outputName='train.hd5',
        txtList='train.list.txt',
        protoTXTPath=PATH_TO_DEPLOY_TXT,
        weightsPath=PATH_TO_WEIGHTS,
        gmm=gmm)

if 'createTestClusters' in STEPS:
    print "Creating HD5 test data for clusters. Loading..."
    with open('testSetPickle.pickle','r') as f:
        dataRows=load(f)
    print "Loaded %d valid rows" % len(dataRows)
    createClusteredData(
        dataRows=dataRows, 
        txtList='test.list.txt',
        outputName='test.hd5', 
        protoTXTPath=PATH_TO_DEPLOY_TXT, 
        weightsPath=PATH_TO_WEIGHTS,
        gmm=gmm)

    print "Finished creating test hd5"

if 'trainClusters' in STEPS:
    for i in range(64):
        trainCluster(i, CLUSTERS_PATH)

DEBUG=True
if 'runTweakTest' in STEPS:
    runTweakTest(gmm)

if 'viewCluster' in STEPS:
    clusterIndex = 0
    clusterPath = os.path.join(CLUSTERS_PATH,str(clusterIndex))
    clusterTrainPath= os.path.join(clusterPath, 'train.hd5')
    import h5py
    f5 = h5py.File(clusterTrainPath)
    print "keys:", f5.keys()
    imgs = f5[f5.keys()[0]]
    lnmrs_05 = f5[f5.keys()[1]]
    print imgs.shape

print "Running time:", timeit.default_timer() - start 

''' STEPS
if __name__=='__main__':
    from qtComm import qtComm
    qtComm('run tweak.py')

'''

