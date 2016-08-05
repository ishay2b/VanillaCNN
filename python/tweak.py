import timeit
import numpy as np
_A = np.array  # A shortcut to creating arrays in command line 
import os
import cv2
import sys
from pickle import load, dump
from zipfile import ZipFile
from urllib import urlretrieve

#Import helper functions
from helpers import *
from DataRow import *


#from matplotlib.pylab import *
FULL_STEPS=[
'createTrainingSet', 
'createGMM', 
'createTrainClusters',
'createTestSetPickle',
'createTestClusters',
'trainCLusters',
'runTest']

STEPS=['', '','runTest']
#STEPS=['createTrainingSet', 'createTrainClusters','createTestClusters','runTest']

start = timeit.default_timer() # total running time 

# Create the MTFL benchmark
if 'createTrainingSet' in STEPS:  
    createTrainingSet()

if 'createGMM' in STEPS:
    createGMM()

#Load GMM
with open('gmm.pickle') as f:
    gmm=load(f)

def createClusteredData(dataRows, outputName, txtList, protoTXTPath, weightsPath):    
    '''
    cluster each data row by nearest neighbor, and write hd5 data to seperate folders 0..63
    Should be called once for train and econd for test data
    '''    
    #Load Vanilla weights 
    predictor = Predictor(protoTXTPath=protoTXTPath, weightsPath=weightsPath)

    #Prepend a vector of 64 vectors
    clusters =[[] for i in range(64)]
    
    for i, dataRow in enumerate(dataRows):
        if i%100 ==0: # Comfort print
            print "Getting feature vector of row:",i
        
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        dataRow40.fvector = predictor.getFeatureVector(image).flatten()
        clusterIndex = findNearestNeigher(gmm, dataRow40.fvector)
        clusters[clusterIndex].append((dataRow40.fvector, lm_0_5))

    dist=[len(c) for c in clusters]
    print "Data distribution:", dist
    #plot(dist); title('Traning clusters number of samples.'); show()
    
    # Create HD5 train data from clussters
    for i in range(64):
        cluster=clusters[i]
        
        vecs=np.array([c[0] for c in cluster])
        landmarks=np.array([c[1] for c in cluster])
                
        clusterPath=os.path.join(CLUSTERS_PATH,str(i))
        if not os.path.isdir(clusterPath):
            os.mkdir(clusterPath)
        dict={
            "ActivationAbs4": vecs,
            "landmarks": landmarks
        }
        writeDictionaryToHD5(dict, os.path.join(clusterPath,outputName), os.path.join(clusterPath,txtList))

if 'createTrainClusters' in STEPS:
    print "Creating train set"
    with open('trainSetMTFL.pickle','r') as f:
        dataRows = load(f)
    createClusteredData(dataRows=dataRows, 
        outputName='train.hd5',
        txtList='train.list.txt',
        protoTXTPath=PATH_TO_DEPLOY_TXT,
        weightsPath=PATH_TO_WEIGHTS)


if 'createTestSetPickle' in STEPS:
    print "Creating test set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d rows from test" % len(dataRowsTest_CSV)
    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print "Original test:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    with open('testSetPickle.pickle','w') as f:
        dump(dataRowsTestValid, f)

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
        weightsPath=PATH_TO_WEIGHTS)

    print "Finished creating test hd5"

if 'trainClusters' in STEPS:
    for i in range(64):
        trainCluster(i, CLUSTERS_PATH)

DEBUG=True
if 'runTest' in STEPS:
    fullyConnected=[OnlyDensePredictor(i) for i in range(64)] # Allocate 64 partitions
    testError=[ErrorAcum() for i in range(64)]
    vanillaTestError=[ErrorAcum() for i in range(64)]

    predictorVanilla = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)

    with open('testSetPickle.pickle') as f:
        dataRowsTestValid=load(f)
        
    print "Loaded ",len(dataRowsTestValid), " valid rows from pickle file."
        
    beginTest = timeit.default_timer()

    for i, dataRow in enumerate(dataRowsTestValid):
        image, lm_0_5 = predictorVanilla.preprocess(dataRow.image, dataRow.landmarks())
        vanillaPrediction, featureVector = predictorVanilla.predictReturnFeatureVectorAsWell(image)
        clusterIndex = findNearestNeigher(gmm, featureVector.flatten())
        prediction = fullyConnected[clusterIndex].predict(featureVector)
        testError[clusterIndex].add(lm_0_5, prediction)
        vanillaTestError[clusterIndex].add(lm_0_5, vanillaPrediction)

        dataRow.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if i%300==0:
            print "run test ",i
            if DEBUG:
                dataRow.prediction = dataRow.inverseScaleAndOffset(dataRow.prediction) # Scale up to the original image scale
                dataRow.show(title=str(i))
            
    for i, err in enumerate(testError):
        print i, "Vanilla Error:",vanillaTestError[i], " tweaked Error:", testError[i]

    print "Time diff",timeit.default_timer()-beginTest 

 

def cleanup(STEPS):
    if 'cleanHD5related' in STEPS:
        pass


print "Running time:", timeit.default_timer() - start 

''' STEPS
a 64 GMM clustring so we can run nearest neighbor 

For each data row:
     preprocess, get FeatureVector Append to gmm input
Calculate GMM
For each data row:
    cluster_index = find nearest neighbor
    dump vector to hd5 cluster bu index

'''
