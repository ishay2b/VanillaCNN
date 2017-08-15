import timeit
import numpy as np
_A = np.array  # A shortcut to creating arrays in command line
import os
import cv2
import sys
from pickle import load, dump
from timeit import default_timer

#Import helper functions
from helpers import *
from DataRow import *
import sys, select, os


#from matplotlib.pylab import *
FULL_STEPS=[
'createTrainingSetPickle',
'createGMM',
'createTrainClusters',
'createTestSetPickle',
'createTestClusters',
'trainCLusters',
'runTweakTest']

STEPS=['augmentClusters']
#STEPS=['createTrainingSetPickle', 'createTrainClusters','createTestClusters','runTweakTest']

start = timeit.default_timer() # total running time

class Stats():
    ''' empty container to reutrn stats
    '''
    pass


def calculateClusterIndex(dataRows):
    clusters =[[] for i in range(64)]

    for i, dataRow in enumerate(dataRows):
        if i%1000 ==0: # Comfort print
            print ("Getting feature vector of row:%d"%i)

        dataRowFaceOnly = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = vanilla_predictor.preprocess(dataRowFaceOnly.image, dataRowFaceOnly.landmarks())
        dataRow.fvector = vanilla_predictor.getFeatureVector(image).flatten() # Save the feature vector to original data fow
        dataRow.clusterIndex = findNearestNeigher(gmm, dataRow.fvector) # Save the cluster index to the original data row
        clusters[dataRow.clusterIndex].append(dataRow)

    dist=[len(c) for c in clusters]
    #plot(dist); title('Traning clusters number of samples.'); show()
    print ("Original data distribution:"+str(dist))
    return clusters



# Create the MTFL benchmark
if 'createTrainingSetPickle' in STEPS:
    createTrainingSetPickle()

if 'createTestSetPickle' in STEPS:
    print ("Creating test set.....")
    dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print ("Finished reading %d rows from test"%len(dataRowsTest_CSV))
    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV, resizeTo=None)

    print ("Original test:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch)
    with open('testSetPickle.pickle','w') as f:
        dump(dataRowsTestValid, f)

#Load Vanilla weights
vanilla_predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
if 'createGMM' in STEPS:
    with open('trainSetMTFL.pickle','r') as f:
        dataRows = load(f)
    gmm=createGMM(vanilla_predictor, dataRows)
    with open('gmm.pickle','w') as f:
        dump(gmm, f)

#Load GMM
start_time = default_timer()
with open('gmm.pickle') as f:
    gmm=load(f)
print ("Finished loading gmm.pickle in %d seconds"%(default_timer()-start_time))

if 'calculateClusterIndexTest' in STEPS: # For debug only, not needed to training
    with open('testSetMTFL.pickle', 'r') as f:
        dataRows = load(f)
    print ("Finished loading %d rows from test data" % len(dataRows))
    train_clusters = calculateClusterIndex(dataRows)
    with open('testSetMTFL.pickle','w') as f:
        dump(dataRows,f)
    print ("Finished resaving train data with feature vectors+cluster index as: trainSetMTFL.pickle")


if 'calculateClusterIndexTrain' in STEPS:
    with open('trainSetMTFL.pickle', 'r') as f:
        dataRows = load(f)
    print ("Finished loading %d rows from train data" % len(dataRows))
    train_clusters = calculateClusterIndex(dataRows)
    with open('trainSetMTFL.pickle','w') as f:
        dump(dataRows,f)
    print ("Finished resaving train data with feature vectors+cluster index as: trainSetMTFL.pickle")

if 'augmentClusters' in STEPS:
    DEBUG = True

    if DEBUG:
        with open('testSetMTFL.pickle', 'r') as f:
            dataRows = load(f)
    else:
        with open('trainSetMTFL.pickle', 'r') as f:
            dataRows = load(f)

    clusters = [[] for i in range(64)]
    stats=Stats()
    stats.WRONG_CLUSTER =0
    stats.AUGMENTET_SUM = 0
    stats.LANDMARKS_OUTOF_RANGE = 0

    for dataRow in dataRows:
        clusters[dataRow.clusterIndex].append(dataRow)

    try:
        for ci, cluster in enumerate(clusters):
            for i in range(len(cluster)-1):
                for j in range (i+1, len(cluster)):
                    A = cluster[i]
                    B = cluster[j]
                    A_T = A.transformedDataRow(B.landmarks()) # Might return a small cropped image due to rotatin
                    if DEBUG:
                        if (len(raw_input("Press Enter to continue...")))>1:
                            raise (KeyboardInterrupt, "User finished")
                    if A_T is None:
                        #Error - landmarks go out of range
                        stats.LANDMARKS_OUTOF_RANGE += 1
                    else:
                        A_T = A_T.copyCroppedByBBox(A_T.fullImageBBox(), resizeTo=(Predictor.SIZE()))

                        processed_image, A_T.lm_0_5 = vanilla_predictor.preprocess(A_T.image, A_T.landmarks())
                        A_T.fvector = vanilla_predictor.getFeatureVector(processed_image).flatten()
                        A_T.clusterIndex = findNearestNeigher(gmm, A_T.fvector)
                        if A_T.clusterIndex != ci:
                            # Error - new image does not belong to the same cluster, therfor rejectet.
                            stats.WRONG_CLUSTER += 1
                            print (vars(stats))
                        else:
                            stats.AUGMENTET_SUM += 1
                            clusters[ci].append(A_T)
    except KeyboardInterrupt:
        print ("KeyboardInterrupt")

    hist_aug = [len(c) for c in clusters]
    print ("clusters sizes pre augmentation:"+str(hist_aug))

    print ("Stats:"+str(vars(stats)))
    write_clusters_hdb5(
        clusters=clusters,
        outputName='train.hd5',
        txtList='train.list.txt')

    print ("Finished writing augmented training data hd5")


if 'createTestClusters' in STEPS:
    print ("Creating HD5 test data for clusters. Loading...")
    with open('testSetPickle.pickle','r') as f:
        dataRows=load(f)
    print ("Loaded %d valid rows"%len(dataRows))
    testClusters = createClusteredData(
        dataRows=dataRows,
        gmm=gmm,
        predictor=vanilla_predictor)
    print ("Finished clustering original test data")
    write_clusters_hdb5(
        testVlusters,
        txtList='test.list.txt',
        outputName='test.hd5')

    print ("Finished creating test hd5")

if 'trainClusters' in STEPS:
    for i in range(2):
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
    print ("keys:"+str(f5.keys()))
    imgs = f5[f5.keys()[0]]
    lnmrs_05 = f5[f5.keys()[1]]
    print (imgs.shape)

print ("Running time: %d seconds"%(default_timer() - start))


''' STEPS
if __name__=='__main__':
    from qtComm import qtComm
    qtComm('run tweak.py')

'''

