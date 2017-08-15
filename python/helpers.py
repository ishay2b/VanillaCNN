import timeit
import numpy as np
_A = np.array  # A shortcut to creating arrays in command line
import os
import cv2
import sys
from pickle import load, dump
from zipfile import ZipFile
try:
    from urllib import urlretrieve #Python2
except:
    from urllib.request import urlretrieve #Python3

def getGitRepFolder():
    from subprocess import Popen, PIPE
    return Popen(['git', 'rev-parse', '--show-toplevel'], stdout=PIPE).communicate()[0].rstrip().decode("utf-8")


#Consts
###########################    PATHS TO SET   ####################
# Either define CAFFE_ROOT in your enviroment variables or set it here
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','~/caffe/distribute')
sys.path.append(CAFFE_ROOT+'/python')  # Add caffe python path so we can import it
import caffe

# Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
#import dlib
#detector=dlib.get_frontal_face_detector() # Load dlib's face detector

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

CLUSTERS_PATH = os.path.join(ROOT, 'clusters')

ORIG_VANILLA_WEIGHTS= os.path.join(CLUSTERS_PATH,'vanillaCNN.caffemodel')

###########################    STEPS TO RUN       ####################
INIT_STEPS =['CreateClusterClassifier']


class OnlyDensePredictor():
    ''' predictor input is the 64*3*3 feature vector and output is 5*2 facial landmarks.
        Currently this is a activation layer with output size 100, cadcaded by another fully connected activation layer with  output size 10
    '''
    def __init__(self, clusterIndex, clustersPath=None):
        self.clusterIndex = clusterIndex
        if clustersPath is None:
            clustersPath =CLUSTERS_PATH

        import caffe, os
        caffe.set_mode_cpu()
        self.net = caffe.Net(
            os.path.join(clustersPath, 'tweak_deploy.prototxt'),
            os.path.join(clustersPath, str(clusterIndex),'best.caffemodel'),
             caffe.TEST)

    def predict(self, feature_vector):
        self.net.blobs['ActivationAbs4'].data[...] = feature_vector
        prediction = self.net.forward()['Dense2'][0]
        return prediction


def emulateBashSource(env_vars='../env_vars.sh'):
	import os, pprint,subprocess

	command = ['bash', '-c', 'source '+env_vars+' && env']

	proc = subprocess.Popen(command, stdout = subprocess.PIPE)
	for line in proc.stdout:
	  (key, _, value) = line.partition("=")
	  os.environ[key] = value.rstrip()

	proc.communicate()

def addPythonModuleToEnvironPath():
	import sys, os
	ppath = os.path.join(getGitRepFolder(),'python')
	if ppath not in sys.path:
		print ("Adding to python path:", ppath)
		sys.path.append(ppath)


def writeDictionaryToHD5(dict, outputPath, setTxtFilePATH):
    ''' Create HD5 data set from a dictonary for caffe from given valid feature vectors per clusster
    if mirror is True, duplicate data by mirroring.
    '''
    import h5py

    setTxtFile = open(setTxtFilePATH, 'w')
    with h5py.File(outputPath, 'w') as T:
        for key in dict:
            T.create_dataset(key, data=dict[key])

    setTxtFile.write(outputPath+"\n")
    setTxtFile.flush()
    setTxtFile.close()


'''    STEPS: Download, create train and test pickle, create train and test hd5 clusters, train, select best snap shot          '''

def downloadTrainingSet():
    from DataRow import createDataRowsFromCSV, getValidWithBBox
    MTFL_LINK = 'http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip'
    MTFL_ZIP = ROOT+"/MTFL.zip"
    if os.path.isfile(MTFL_ZIP):
        print ("MTFL.zip already downloaded")
    else:
        print ("Downloading:"+MTFL_ZIP+" from url:"+MTFL_LINK+".....")
        urlretrieve(MTFL_LINK, MTFL_ZIP)
        print ("Finished download. Extracting file.....")
        with ZipFile(MTFL_ZIP) as f:
            f.extractall(ROOT+'/data')
            print ("Done extracting MTFL")
            f.close()

def downloadAFW():
    theurl = 'https://www.ics.uci.edu/~xzhu/face/AFW.zip'
    filename = ROOT+'/AFW.zip'
    if os.path.isfile(filename):
        print ("AFW.zip already downloaded")
    else:
        print ("Downloading "+theurl + " .....")
        name, hdrs = urlretrieve(theurl, filename)
        print ("Finished downloading AFW. Extracting AFW zip file......")
        folderPATH = ROOT+'/data'
        with ZipFile(filename) as theOpenedFile:
            theOpenedFile.extractall(folderPATH)
            theOpenedFile.close()

def downloadAFLW():
    theurl='http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip'
    filename = ROOT+'/AFLW.zip'
    if os.path.isfile(filename):
        print ("AFLW.zip training data already downloaded")
    else:
        print ("Downloading "+theurl + " .....")
        name, hdrs = urlretrieve(theurl, filename)
        print ("Finished downloading AFLW. Extracting zip data...")
        folderPATH = ROOT+'/data'
        with ZipFile(filename) as theOpenedFile:
            theOpenedFile.extractall(folderPATH)
            theOpenedFile.close()
        print ("Done extracting AFW zip folder")


def createTrainingSetPickle():
    #from DataRow import *

    downloadTrainingSet()
    AFLW_PATH = os.path.join(ROOT,'data')
    CSV_MTFL = os.path.join(AFLW_PATH,'training.txt')
    print ("Parsine training data CSV, May take some minutes.....")
    dataRowsMTFL_CSV  = createDataRowsFromCSV(CSV_MTFL , DataRow.DataRowFromMTFL, AFLW_PATH)
    print ("Finished reading %d rows from train" % len(dataRowsMTFL_CSV))
    dataRowsMTFLValid,R = getValidWithBBox(dataRowsMTFL_CSV, resizeTo=_A([40., 40.]))
    print ("Original train:",len(dataRowsMTFL_CSV), "Valid Rows:", len(dataRowsMTFLValid), " No faces at all", R.noFacesAtAll, " Illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch)
    with open('trainSetMTFL.pickle','w') as f:
        dump(dataRowsMTFLValid,f)
    print ("Finished dumping to trainSetMTFL.pickle")
    dataRowsMTFL_CSV=[]


def createGMM(predictor, dataRows):
    from sklearn import mixture

    gmmStartTime = timeit.default_timer()
    gmmInput=[]

    for i, dataRow in enumerate(dataRows):
        if i%100 ==0:
            print ("Getting feature vector of row:",i)

        cropped = dataRow.copyCroppedByBBox(dataRow.fbbox, resizeTo=predictor.SIZE()) # Get face only and resize to 40x40
        image, lm_0_5 = predictor.preprocess(cropped.image, cropped.landmarks()) # Reduce mean, divide by std.
        fvector = predictor.getFeatureVector(image) # run only paritail network, output shape is (1, 64, 3, 3)
        gmmInput.append(fvector.flatten()) # append to gmm input, treat as a flat 576 float vector.

    print ("Extracted ",len(gmmInput)," feature vector of shape:", fvector.shape, " Building GMM will take some time...")
    #Calculate GMM
    gmix = mixture.GMM(n_components=64, covariance_type='full')
    gmix.fit(gmmInput)
    print ("createGMM: run time ",timeit.default_timer() - gmmStartTime )
    return gmix.means_


def createClusteredData(dataRows, gmm, predictor):
    '''
    cluster each data row by nearest neighbor, and append the data row with feature vector + cluster index
    Should be called once for train and econd for test data
    '''
    #Prepend a vector of 64 vectors
    clusters =[[] for i in range(64)]

    for i, dataRow in enumerate(dataRows):
        if i%100 ==0: # Comfort print
            print ("Getting feature vector of row:",i)

        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        dataRow.fvector = predictor.getFeatureVector(image).flatten() # Save the feature vector to original data fow
        dataRow.clusterIndex = findNearestNeigher(gmm, dataRow40.fvector) # Save the cluster index to the original data row
        clusters[dataRow.clusterIndex].append((dataRow.fvector, lm_0_5))

    dist=[len(c) for c in clusters]
    #plot(dist); title('Traning clusters number of samples.'); show()
    print ("Original data distribution:", dist)
    return clusters


def write_clusters_hdb5(clusters, outputName, txtList):
    # Create HD5 train data from clussters
    for i in range(64):
        cluster=clusters[i]

        vecs=np.array([dataRow.fvector for dataRow in cluster])
        landmarks=np.array([dataRow.landmarks_0_5() for dataRow in cluster]) # write scaled landmarks -0.5..+0.5 to hdf

        clusterPath=os.path.join(CLUSTERS_PATH,str(i))
        if not os.path.isdir(clusterPath):
            os.mkdir(clusterPath)
        dict={
            "ActivationAbs4": vecs,
            "landmarks": landmarks
        }
        writeDictionaryToHD5(dict, os.path.join(clusterPath,outputName), os.path.join(clusterPath,txtList))



def parseLog(logFilePath):
    import re
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
                xAxis.append(int(xresult.groups(1)[0]))

            yresult = regexLos.search(line)
            if (yresult):
                yAxis.append(float(yresult.groups(1)[0]))

    mnSize = min(len(xAxis),len(yAxis))
    skip = 0

    xAxis = xAxis[skip:mnSize]
    yAxis = yAxis[skip:mnSize]
    return xAxis, yAxis


def createBest(clustersPath, pathToOrig=ORIG_VANILLA_WEIGHTS):
    '''
    Assume snapshot intervel is synced with iteration loss plot we can get the best snapshot
    pathToOrig is needed if the loss only got worse, we use original set, debug mode.
    '''
    import os, shutil
    from glob import glob

    xAxis, yAxis = parseLog(os.path.join(clustersPath,'clusterLog.txt'))
    print ("Parsed loss enteries:", len(xAxis) ,len(yAxis))
    minIndex=np.argmin(yAxis)
    x,y= xAxis[minIndex], yAxis[minIndex]

    print ("Min loss error found at (x,y):", x, y)
    source = os.path.join(clustersPath, 'snap_iter_%d.caffemodel' % x)
    if minIndex==0:
        print ("Error always went up!!!. The first initial snapshot was better, using it. Need to know why this happened. Choosing latest")
        files = glob("snap_iter*.caffemodel")
        files.sort(key=os.path.getmtime)
        source = files[0]

    target = os.path.join(clustersPath, 'best.caffemodel')
    print ("Copying:", source, "  TO: ", target)
    shutil.copy(source,target)
    return xAxis, yAxis


def findNearestNeigher(gmm, v):
    '''return nearest neighbor index for vector v in gmm'''
    import numpy as np
    err=np.sum((gmm-v)**2,axis=1)
    minIndex= np.argmin(err)
    return minIndex



def trainCluster(clusterIndex, pathToClusters):
    import sys, os
    from subprocess import call
    import caffe
    from glob import glob
    emulateBashSource() # Make sure to load all environ needed in env_vars.sh

    prevDir=os.getcwd() # we will change directory and return to this
    os.chdir(os.path.join(pathToClusters,str(clusterIndex)))
    pathToTrainProto    = os.path.join(pathToClusters,'tweak_train.prototxt')
    pathToWeights   = os.path.join(pathToClusters,'vanillaCNN.caffemodel')
    pathToSolver        = os.path.join(pathToClusters,'tweak_adam_solver.prototxt')

    '''
    orig_std=sys.stdout
    sys.stdout=open('clusterLog.txt','w')
    solver = caffe.get_solver(pathToSolver)
    net=caffe.Net(pathToTrainProto, pathToWeights, caffe.TRAIN)

    solver.solve()
    sys.stdout.close()
    sys.stdout=orig_std
    os.chdir(prevDir)
    return
    '''

    #Get the caffe.bin exe, Assume to be define in env_vars.sh
    caffeExe=os.environ.get('CAFFE_EXE','/Users/ishay/caffe/distribute/bin/caffe.bin')

    cmd=[caffeExe, 'train',
        '--solver', pathToSolver,
        '--weights', pathToWeights
        ]

    print (cmd)

    #call caffe.bin with the params and redirect both output to clusterLog.txt
    with open('clusterLog.txt', "w") as outfile:
        ret=call(cmd, stdout=outfile, stderr=outfile)

    #Parse log, find min error, create a copy of min snapshot weights named best.caffemodel
    createBest(os.getcwd())

    # Clean up snap states and model
    for snap in glob('snap_iter*'):
        os.remove(snap)

    #return current working dir as where started
    os.chdir(prevDir)



def runTweakTest(gmm, DEBUG=False):
    from DataRow import ErrorAcum, Predictor
    fullyConnected=[OnlyDensePredictor(i) for i in range(64)] # Allocate 64 partitions
    testError=[ErrorAcum() for i in range(64)]
    vanillaTestError=[ErrorAcum() for i in range(64)]

    predictorVanilla = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)

    with open('testSetPickle.pickle') as f:
        dataRowsTestValid=load(f)

    print ("Loaded ",len(dataRowsTestValid), " valid rows from pickle file.")

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
            print ("run test ",i)
            if DEBUG:
                dataRow.prediction = dataRow.inverseScaleAndOffset(dataRow.prediction) # Scale up to the original image scale
                dataRow.show(title=str(i))

    for i, err in enumerate(testError):
        print (i, "Vanilla Error:",vanillaTestError[i], " tweaked Error:", testError[i])

    print ("Time diff running tweak test",timeit.default_timer()-beginTest)


def matrixPlot(vec, title_=''):
    ''' plot a vector of images in a matrix of sqrt(len(vec)) ^ 2
    '''
    figure()
    title(title_)
    z=0
    rows =int(ceil(len(vec)**0.5))
    cols =int(floor(len(vec)/rows))
    for i in range(rows):
        for j in range(cols):
            if z<len(vec):
                subplot(rows, cols, z+1)
                imshow(cv2.cvtColor(vec[z], cv2.COLOR_BGR2RGB))
                axis("off")
            z += 1

