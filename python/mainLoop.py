
import numpy as np
from mercurial.templater import if_
_A = np.array  # A shortcut to creating arrays in command line 
import os
import cv2
import sys
from pickle import load, dump
from zipfile import ZipFile
from urllib import urlretrieve

#Import helper functions
from DataRow import DataRow, ErrorAcum, Predictor, getGitRepFolder, createDataRowsFromCSV, getValidWithBBox, writeHD5

###########################    PATHS TO SET   ####################
# Either define CAFFE_ROOT in your enviroment variables or set it here
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','~/caffe/distribute')
sys.path.append(CAFFE_ROOT+'/python')  # Add caffe python path so we can import it
import caffe

# Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
import dlib
detector=dlib.get_frontal_face_detector() # Load dlib's face detector



ROOT = getGitRepFolder()  # ROOT is the git root folder .
sys.path.append(os.path.join(ROOT, 'python'))  # Assume git root directory
DATA_PATH = os.path.join(ROOT, 'data')
CSV_TEST  = os.path.join(ROOT, 'data', 'testImageList.txt')
CSV_TRAIN = os.path.join(ROOT, 'data', 'trainImageList.txt')

MEAN_TRAIN_SET = cv2.imread(os.path.join(ROOT, 'trainMean.png')).astype('f4')
STD_TRAIN_SET  = cv2.imread(os.path.join(ROOT, 'trainSTD.png')).astype('f4')

AFW_DATA_PATH = os.path.join(ROOT, 'data', 'testimages')
AFW_MAT_PATH = os.path.join(ROOT, 'data', 'anno-v7.mat')

PATH_TO_WEIGHTS  = os.path.join(ROOT, 'ZOO', 'vanillaCNN.caffemodel')
PATH_TO_DEPLOY_TXT = os.path.join(ROOT, 'ZOO', 'vanilla_deploy.prototxt')

MEAN_TRAIN_SET = cv2.imread(os.path.join(ROOT, 'trainMean.png')).astype('f4')
STD_TRAIN_SET  = cv2.imread(os.path.join(ROOT, 'trainSTD.png')).astype('f4')

detector=dlib.get_frontal_face_detector()

###########################    STEPS TO RUN       ####################
AFW_STEPS =['downloadAFW', 
            'createAFW_TestSet',
            'testAFW_TestSet'] # Steps needed for AFW

AFLW_STEPS=['downloadALFW', 
            'testSetHD5', 
            'testSetPickle', 
            'trainSetHD5', 
            'createAFLW_TestSet', 
            'testAFLW_TestSet', 
            'testErrorMini'] # Steps needed for AFLW

STEPS =['createAFLW_TestSet'] # AFLW_STEPS+AFW_STEPS # Run AFLW and AFW steps

##########################################    SCRIPT STEPS       ##################################################

if 'downloadAFW' in STEPS:
    theurl = 'https://www.ics.uci.edu/~xzhu/face/AFW.zip'
    filename = ROOT+'/AFW.zip'
    if os.path.isfile(filename):
        print "AFW.zip already downloaded"
    else:
        print "Downloading "+theurl + " ....."
        name, hdrs = urlretrieve(theurl, filename)
        print "Finished downloading AFW....."
        
    print "Extracting AFW zip file......"
    folderPATH = ROOT+'/data'
    with ZipFile(filename) as theOpenedFile:
        theOpenedFile.extractall(folderPATH)
        theOpenedFile.close()

if 'downloadAFLW' in STEPS:
    theurl='http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip'
    filename = ROOT+'/AFLW.zip'
    if os.path.isfile(filename):
        print "AFLW.zip training data already downloaded"
    else:
        print "Downloading "+theurl + " ....."
        name, hdrs = urlretrieve(theurl, filename)
        print "Finished downloading AFLW....."

    print 'Extracting zip data...'
    folderPATH = ROOT+'/data'
    with ZipFile(filename) as theOpenedFile:
        theOpenedFile.extractall(folderPATH)
        theOpenedFile.close()
    print "Done extracting AFW zip folder"

if 'testSetHD5' in STEPS or 'testSetPickle' in STEPS:
    print "Creating test set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d rows from test" % len(dataRowsTest_CSV)
    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print "Original test:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    if 'testSetPickle' in STEPS:
        with open('testSet.pickle','w') as f:
            dump(dataRowsTestValid,f)
        print "Finished dumping to testSet.pickle"
        # Also save mini test set for debug
        with open('testSetMini.pickle','w') as f:
            dump(dataRowsTestValid[0:10],f)
        print "Finished dumping to testSetMini.pickle"
        
        
    if 'testSetHD5' in STEPS:
        writeHD5(dataRowsTestValid, ROOT+'/caffeData/hd5/test.hd5', ROOT+'/caffeData/test.txt', MEAN_TRAIN_SET, STD_TRAIN_SET)
        print "Finished writing test to caffeData/test.txt"

 
DEBUG = True
if 'testErrorMini' in STEPS:
    with open('testSetMini.pickle','r') as f:
        dataRowsTrainValid = load(f)
        
    testErrorMini=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsTrainValid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox).copyMirrored()
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorMini.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()
            if i>40:
                print "Debug breaked after %d rows:" % i
            

    print "Test Error mini:", testErrorMini

  
  
if 'trainSetHD5' in STEPS:
    dataRowsTrain_CSV = createDataRowsFromCSV(CSV_TRAIN, DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d rows from training data. Parsing BBox...." % len(dataRowsTrain_CSV)
    dataRowsTrainValid,R = getValidWithBBox(dataRowsTrain_CSV)
    print "Original train:",len(dataRowsTrain_CSV), "Valid Rows:", len(dataRowsTrainValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    dataRowsTrain_CSV=[]  # remove from memory
    
    writeHD5(dataRowsTrainValid, ROOT+'/caffeData/hd5/train.hd5', ROOT+'/caffeData/train.txt', MEAN_TRAIN_SET, STD_TRAIN_SET ,mirror=True)
    print "Finished writing train to caffeData/train.txt"
    

# Run the same caffe test set using python
DEBUG = False  # Set this to true if you wish to plot the images
if 'testError' in STEPS:
    with open('testSet.pickle','r') as f:
        dataRowsTrainValid = load(f)
        
    testError=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsTrainValid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm40 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testError.add(lm40, prediction)
        dataRow40.prediction = (prediction+0.5)*40.
        
        if DEBUG and i%40 ==0:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction)
            dataRow.show()
            break
        
            
    print "Test Error:", testError

  
    
# AFW test - Make the pickle data set
if 'createAFW_TestSet' in STEPS:
    print "Parsing AFW anno-v7.mat ....."
    from scipy.io import loadmat
    annotaions = loadmat(AFW_MAT_PATH)['anno']
    dataRowsAFW = []
        
    for anno in annotaions:
        dataRow = DataRow.DataRowFromAFW(anno, AFW_DATA_PATH)
        if dataRow is not None:
            dataRowsAFW.append(dataRow)
    print "Finished parsing anno-v7.mat with total rows:", len(dataRowsAFW)
    annotaions = None  # remove from memory
    
    dataRowsAFW_Valid, R=getValidWithBBox(dataRowsAFW)
    print "Original AFW:",len(dataRowsAFW), "Valid Rows:", len(dataRowsAFW_Valid), " No faces at all", R.noFacesAtAll, " illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch
    dataRowsAFW = None  # remove from Memory
    with open('afwTestSet.pickle','w') as f:
        dump(dataRowsAFW_Valid, f)
        print "Data saved to afwTestSet.pickle"
    
    
DEBUG = False
if 'testAFW_TestSet' in STEPS:
    with open('afwTestSet.pickle','r') as f:
        dataRowsAFW_Valid = load(f)

    testErrorAFW=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsAFW_Valid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorAFW.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()
            

    print "Test Error AFW:", testErrorAFW

    

# Create the MTFL benchmark
if 'createAFLW_TestSet' in STEPS:  
    MTFL_LINK = 'http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip'
    MTFL_ZIP = ROOT+"/MTFL.zip"
    if os.path.isfile(MTFL_ZIP):
        print "MTFL.zip already downloaded"
    else:
        print "Downloading:"+MTFL_ZIP+" from url:"+MTFL_LINK+"....."
        urlretrieve(MTFL_LINK, MTFL_ZIP)
        print "Finished download. Extracting file....."
        with ZipFile(MTFL_ZIP) as f:
            f.extractall(ROOT+'/data')
            print "Done extracting MTFL"
            f.close()
            
    AFLW_PATH = os.path.join(ROOT,'data')
    CSV_MTFL = os.path.join(AFLW_PATH,'testing.txt')
    dataRowsMTFL_CSV  = createDataRowsFromCSV(CSV_MTFL , DataRow.DataRowFromMTFL, AFLW_PATH)
    print "Finished reading %d rows from test" % len(dataRowsMTFL_CSV)
    dataRowsMTFLValid,R = getValidWithBBox(dataRowsMTFL_CSV)
    print "Original test:",len(dataRowsMTFL_CSV), "Valid Rows:", len(dataRowsMTFLValid), " No faces at all", R.noFacesAtAll, " Illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch
    with open('testSetMTFL.pickle','w') as f:
        dump(dataRowsMTFLValid,f)
    print "Finished dumping to testSetMTFL.pickle"        


# Run AFLW benchmark
DEBUG = False
if 'testAFLW_TestSet' in STEPS:
    print "Running AFLW benchmark........."
    with open('testSetMTFL.pickle','r') as f:
        dataRowsAFW_Valid = load(f)
    print "%d rows in AFLW benchmark ....." % len(dataRowsAFW_Valid)
    testErrorAFLW=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsAFW_Valid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorAFLW.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG and i%40 == 0:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()


    print "Test Error AFLW:", testErrorAFLW


if 'findSnapshotWithMinError' in STEPS:
    pass
