# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:30:29 2015

@author: Ishay Tubi
"""

import os
import cv2
import numpy as np
import sys
import csv

from helpers import *

def mse_normlized(groundTruth, pred):
    delX = groundTruth[0]-groundTruth[2] 
    delY = groundTruth[1]-groundTruth[3] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    diff = (pred-groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 




class RetVal:
    pass  ## A generic class to return multiple values without a need for a dictionary.

def createDataRowsFromCSV(csvFilePath, csvParseFunc, DATA_PATH, limit = sys.maxint):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0 
    
    with open(csvFilePath, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            d = csvParseFunc(row, DATA_PATH)
            if d is not None:
                data.append(d)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
    return data

def getValidWithBBox(dataRows):
    ''' Returns a list of valid DataRow of a given list of dataRows 
    '''
    import dlib
    R=RetVal()
    
    R.outsideLandmarks = 0 
    R.noImages = 0 
    R.noFacesAtAll = 0 
    R.couldNotMatch = 0
    detector=dlib.get_frontal_face_detector()

    validRow=[]
    for dataRow in dataRows:
        if dataRow.image is None or len(dataRow.image)==0:
            R.noImages += 1
        lmd_xy = dataRow.landmarks().reshape([-1,2])
        left,  top = lmd_xy.min( axis=0 )
        right, bot = lmd_xy.max( axis=0 )
                
        dets = detector( np.array(dataRow.image, dtype = 'uint8' ) );
        
        det_bbox = None  # the valid bbox if found 
    
        for det in dets:
            det_box = BBox.BBoxFromLTRB(det.left(), det.top(), det.right(), det.bottom())
            
            # Does all landmarks fit into this box?
            if top >= det_box.top and bot<= det_box.bottom and left>=det_box.left and right<=det_box.right:
                det_bbox = det_box  
                    
        if det_bbox is None:
            if len(dets)>0:
                R.couldNotMatch += 1  # For statistics, dlib found faces but they did not match our landmarks.
            else:
                R.noFacesAtAll += 1  # dlib found 0 faces.
        else:
            dataRow.fbbox = det_bbox  # Save the bbox to the data row
            if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[0] or det_bbox.bottom>dataRow.image.shape[1]:
                R.outsideLandmarks += 1  # Saftey check, make sure nothing goes out of bound.
            else:
                validRow.append(dataRow.copyCroppedByBBox(dataRow.fbbox))  
    
    
    return validRow,R 
        
def writeHD5(dataRows, outputPath, setTxtFilePATH, meanTrainSet, stdTrainSet , IMAGE_SIZE=40, mirror=False):
    ''' Create HD5 data set for caffe from given valid data rows.
    if mirror is True, duplicate data by mirroring. 
    ''' 
    from numpy import zeros
    import h5py
    
    if mirror:
        BATCH_SIZE = len(dataRows) *2
    else:
        BATCH_SIZE = len(dataRows) 

    HD5Images = zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
    HD5Landmarks = zeros([BATCH_SIZE, 10], dtype='float32')
    #prefix  = os.path.join(ROOT, 'caffeData', 'hd5', 'train')
    setTxtFile = open(setTxtFilePATH, 'w')

        
    i = 0 
    
    for dataRowOrig in dataRows:
        if i % 1000 == 0 or i >= BATCH_SIZE-1:
            print "Processing row %d " % (i+1) 
            
        if not hasattr(dataRowOrig, 'fbbox'):
            print "Warning, no fbbox"
            continue
        
        dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox)  # Get a cropped scale copy of the data row
        scaledLM = dataRow.landmarksScaledMinus05_plus05() 
        image = dataRow.image.astype('f4')
        image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
        
        HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
        HD5Landmarks[i,:] = scaledLM
        i+=1
        
        if mirror:
            dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox).copyMirrored()  # Get a cropped scale copy of the data row
            scaledLM = dataRow.landmarksScaledMinus05_plus05() 
            image = dataRow.image.astype('f4')
            image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
            
            HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Landmarks[i,:] = scaledLM
            i+=1
        
        
    with h5py.File(outputPath, 'w') as T:
        T.create_dataset("X", data=HD5Images)
        T.create_dataset("landmarks", data=HD5Landmarks)

    setTxtFile.write(outputPath+"\n")
    setTxtFile.flush()
    setTxtFile.close()
    
    
    
  



class ErrorAcum:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(5, dtype ='f4')
        self.itemsCounter = 0
        self.failureCounter = 0
        
    def __repr__(self):
        return '%f mean error, %d items, %d failures  %f precent' % (self.meanError().mean()*100, self.itemsCounter, self.failureCounter, float(self.failureCounter)/self.itemsCounter if self.itemsCounter>0 else 0)
        
        
    def add(self, groundTruth, pred):
        normlized = mse_normlized(groundTruth, pred)
        self.errorPerLandmark += normlized
        self.itemsCounter +=1
        if normlized.mean() > 0.1: 
            # Count error above 10% as failure
            self.failureCounter +=1

    def meanError(self):
        if self.itemsCounter > 0:
            return self.errorPerLandmark/self.itemsCounter
        else:
            return self.errorPerLandmark

    def __add__(self, x):
        ret = ErrorAcum()
        ret.errorPerLandmark = self.errorPerLandmark + x.errorPerLandmark
        ret.itemsCounter    = self.itemsCounter + x.itemsCounter
        ret.failureCounter  = self.failureCounter + x.failureCounter        
        return ret
        
    def plot(self):
        from matplotlib.pylab import show, plot, stem
        pass


class BBox:  # Bounding box
    
    @staticmethod
    def BBoxFromLTRB(l, t, r, b):
        return BBox(l, t, r, b)
    
    @staticmethod
    def BBoxFromXYWH_array(xywh):
        return BBox(xywh[0], xywh[1], +xywh[0]+xywh[2], xywh[1]+xywh[3])
    
    @staticmethod
    def BBoxFromXYWH(x,y,w,h):
        return BBox(x,y, x+w, y+h)
    
    def top_left(self, dtype=None):
        if dtype is None:
            return (self.top, self.left)
        elif dtype=='int':
            return (int(self.top), int(self.left))
        elif dtype=='float':
            return (float(self.top), float(self.left))
        else:
            print "Error type requested, only int, float of None"
    
    def left_top(self, dtype=None):        
        if dtype is None:
            return (self.left, self.top)
        elif dtype=='int':
            return (int(self.left), int(self.top))
        elif dtype=='float':
            return (float(self.left), float(self.top))
        else:
            print "Error type requested, only int, float of None"

    def bottom_right(self, dtype=None):
        if dtype is None:
            return (self.bottom, self.right)
        elif dtype=='int':
            return (int(self.bottom), int(self.right))
        elif dtype=='float':
            return (float(self.bottom), float(self.right))
        else:
            print "Error type requested, only int, float of None"
    
    def right_top(self, dtype=None):
        if dtype is None:
            return (self.right, self.top)
        elif dtype=='int':
            return (int(self.right), int(self.top))
        elif dtype=='float':
            return (float(self.right), float(self.top))
        else:
            print "Error type requested, only int, float of None"
    
    def relaxed(self, clip ,relax=3):  #@Unused
        from numpy import array
        _A = array
        maxWidth, maxHeight =  clip[0], clip[1]
        
        nw, nh = self.size()*(1+relax)*.5       
        center = self.center()
        offset=_A([nw,nh])
        lefttop = center - offset
        rightbot= center + offset 
         
        self.left, self.top  = int( max( 0, lefttop[0] ) ), int( max( 0, lefttop[1]) )
        self.right, self.bottom = int( min( rightbot[0], maxWidth ) ), int( min( rightbot[1], maxHeight ) )
        return self

    def clip(self, maxRight, maxBottom):
        self.left = max(self.left, 0)
        self.top = max(self.top, 0)
        self.right = min(self.right, maxRight)
        self.bottom = min(self.bottom, maxBottom)
        
    def size(self):
        from numpy import  array
        return array([self.width(), self.height()])
     
    def center(self):
        from numpy import  array
        return array([(self.left+self.right)/2, (self.top+self.bottom)/2])
                
    def __init__(self,left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        
    def width(self):
        return self.right - self.left
        
    def height(self):
        return self.bottom - self.top
        
    def xywh(self):
        return self.left, self.top, self.width(), self.height()
        
    def offset(self, x, y):
        self.left += x 
        self.right += x
        self.top += y 
        self.bottom += y
         
    def scale(self, rx, ry):
        self.left *= rx 
        self.right *= rx
        self.top *= ry 
        self.bottom *= ry
                        
    def __repr__(self):
        return 'left(%.1f), top(%.1f), right(%.1f), bottom(%.1f) w(%d) h(%d)' % (self.left, self.top, self.right, self.bottom,self.width(), self.height())

    def makeInt(self):
        self.left    = int(self.left)
        self.top     = int(self.top)
        self.right   = int(self.right)
        self.bottom  = int(self.bottom)
        return self



class DataRow:
    global TrainSetMean
    global TrainSetSTD
    
    IMAGE_SIZE = 40
    def __init__(self, path='', leftEye=(0, 0, ), rightEye=(0, 0), middle=(0, 0), leftMouth=(0, 0), rightMouth=(0, 0)):
        self.image = cv2.imread(path)
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.leftMouth = leftMouth
        self.rightMouth = rightMouth
        self.middle = middle
        self.name = os.path.split(path)[-1]
        self.sx = 1.
        self.sy = 1.
        self.offsetX = 0.
        self.offsetY = 0.

    def __repr__(self):
        return '{} le:{},{} re:{},{} nose:{},{}, lm:{},{} rm:{},{}'.format(
            self.name,
            self.leftEye[0], self.leftEye[1],
            self.rightEye[0], self.rightEye[1],
            self.middle[0], self.middle[1],
            self.leftMouth[0], self.leftMouth[1],
            self.rightMouth[0], self.rightMouth[1]
            )

    def setLandmarks(self,landMarks):
        """
        @landMarks : np.array
        set the landmarks from array
        """
        self.leftEye = landMarks[0:2]
        self.rightEye = landMarks[2:4]
        self.middle = landMarks[4:6]
        self.leftMouth = landMarks[6:8]
        self.rightMouth = landMarks[8:10]
        
        
    def landmarks(self):
        # return numpy float array with ordered values
        stright = [
            self.leftEye[0],
            self.leftEye[1],
            self.rightEye[0],
            self.rightEye[1],
            self.middle[0],
            self.middle[1],
            self.leftMouth[0],
            self.leftMouth[1],
            self.rightMouth[0],
            self.rightMouth[1]]

        return np.array(stright, dtype='f4')

    def landmarksScaledMinus05_plus05(self):
        # return numpy float array with ordered values
        return self.landmarks().astype('f4')/40. - 0.5
        
    def scale(self, sx, sy):
        self.sx *= sx
        self.sy *= sy

        self.leftEye = (self.leftEye[0]*sx, self.leftEye[1]*sy)
        self.rightEye = (self.rightEye[0]*sx, self.rightEye[1]*sy)
        self.middle = (self.middle[0]*sx, self.middle[1]*sy)
        self.leftMouth = (self.leftMouth[0]*sx, self.leftMouth[1]*sy)
        self.rightMouth = (self.rightMouth[0]*sx, self.rightMouth[1]*sy)
        
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1, 2)*[sx, sy]

        return self

    def offsetCropped(self, offset=(0., 0.)):
        """ given the cropped values - offset the positions by offset
        """
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]

        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1,2)-offset


        self.leftEye = (self.leftEye[0]-offset[0], self.leftEye[1]-offset[1])
        self.rightEye = (self.rightEye[0]-offset[0], self.rightEye[1]-offset[1])
        self.middle = (self.middle[0]-offset[0], self.middle[1]-offset[1])
        self.leftMouth = (self.leftMouth[0]-offset[0], self.leftMouth[1]-offset[1])
        self.rightMouth = (self.rightMouth[0]-offset[0], self.rightMouth[1]-offset[1])
        return self

    def inverseScaleAndOffset(self, landmarks):
        """ computes the inverse scale and offset of input data according to the inverse scale factor and inverse offset factor
        """
        from numpy import array; _A = array ; # Shothand 
        
        ret = _A(landmarks.reshape(-1,2)) *_A([1./self.sx, 1./self.sy])
        ret += _A([-self.offsetX, -self.offsetY])
        return ret

    @staticmethod
    def DataRowFromNameBoxInterlaved(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        d.leftEye = (float(row[5]), float(row[6]))
        d.rightEye = (float(row[7]), float(row[8]))
        d.middle = (float(row[9]), float(row[10]))
        d.leftMouth = (float(row[11]), float(row[12]))
        d.rightMouth = (float(row[13]), float(row[14]))

        return d

    @staticmethod
    def DataRowFromMTFL(row, root=''):
        '''
        --x1...x5,y1...y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
        '''
        d = DataRow()
        if len(row[0]) <= 1:
            # bug in the files, it has spaces seperating them, skip it
            row=row[1:]
            
        if len(row)<10:
            print 'error parsing ', row
            return None

        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        
        if d.image is None:
            print 'Error reading image', d.path
            return None
        
        d.leftEye = (float(row[1]), float(row[6]))
        d.rightEye = (float(row[2]), float(row[7]))
        d.middle = (float(row[3]), float(row[8]))
        d.leftMouth = (float(row[4]), float(row[9]))
        d.rightMouth = (float(row[5]), float(row[10]))
        return d

    @staticmethod
    def DataRowFromAFW(anno, root=''): # Assume data comming from parsed anno-v7.mat file.
        name = str(anno[0][0])
#        bbox = anno[1][0][0]
#        yaw, pitch, roll = anno[2][0][0][0]
        lm = anno[3][0][0]  # 6 landmarks

        if np.isnan(lm).any():
            return None  # Fail

        d = DataRow()
        d.path = os.path.join(root, name).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        d.leftEye = (float(lm[0][0]), float(lm[0][1]))
        d.rightEye = (float(lm[1][0]), float(lm[1][1]))
        d.middle = (float(lm[2][0]), float(lm[2][1]))
        d.leftMouth = (float(lm[3][0]), float(lm[3][1]))
        # skip point 4 middle mouth - We take 0 left eye, 1 right eye, 2 nose, 3 left mouth, 5 right mouth
        d.rightMouth = (float(lm[5][0]), float(lm[5][1]))

        return d

    @staticmethod
    def DataRowFromPrediction(p, path='', image=None):
        d = DataRow(path)        
        p = (p+0.5)*40.  # scale from -0.5..+0.5 to 0..40
        
        d.leftEye = (p[0], p[1])
        d.rightEye = (p[2], p[3])
        d.middle = (p[4], p[5])
        d.leftMouth = (p[6], p[7])
        d.rightMouth = (p[8], p[9])

        return d

    def drawLandmarks(self, r=2, color=255, other=None, title=None):
        M = self.image
        if hasattr(self, 'prediction'):
            for x,y in self.prediction.reshape(-1,2):
                cv2.circle(M, (int(x), int(y)), r, (0,200,0), -1)            

        cv2.circle(M, (int(self.leftEye[0]), int(self.leftEye[1])), r, color, -1)
        cv2.circle(M, (int(self.rightEye[0]), int(self.rightEye[1])), r, color, -1)
        cv2.circle(M, (int(self.leftMouth[0]), int(self.leftMouth[1])), r, color, -1)
        cv2.circle(M, (int(self.rightMouth[0]), int(self.rightMouth[1])), r, color, -1)
        cv2.circle(M, (int(self.middle[0]), int(self.middle[1])), r, color, -1)
        if hasattr(self, 'fbbox'):
            cv2.rectangle(M, self.fbbox.top_left(dtype='int'), self.fbbox.bottom_right(dtype='int'), color)
        return M

    def show(self, r=2, color=255, other=None, title=None):
        M = self.drawLandmarks(r, color, other, title)
        if title is None:
            title = self.name
        cv2.imshow(title, M)

        return M
        
    def makeInt(self):
        self.leftEye    = (int(self.leftEye[0]), int(self.leftEye[1]))
        self.rightEye   = (int(self.rightEye[0]), int(self.rightEye[1]))
        self.middle     = (int(self.middle[0]), int(self.middle[1]))
        self.leftMouth  = (int(self.leftMouth[0]), int(self.leftMouth[1]))
        self.rightMouth = (int(self.rightMouth[0]), int(self.rightMouth[1]))
        return self        
         
    def copyCroppedByBBox(self,fbbox, siz=np.array([40,40])):
        """
        @ fbbox : BBox
        Returns a copy with cropped, scaled to size
        """        
        
        fbbox.makeInt() # assume BBox class
        if fbbox.width()<10 or fbbox.height()<10:
            print "Invalid bbox size:",fbbox
            return None
            
        faceOnly = self.image[fbbox.top : fbbox.bottom, fbbox.left:fbbox.right, :]
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))    
        scaled.setLandmarks(self.landmarks())        
        """ @scaled: DataRow """
        scaled.offsetCropped(fbbox.left_top()) # offset the landmarks
        rx, ry = siz.astype('f4')/faceOnly.shape[:2]
        scaled.scale(rx, ry)
        scaled.fbbox=BBox.BBoxFromLTRB(0, 0, int(siz[0]), int(siz[1]))
        return scaled        
        
    def copyMirrored(self):
        '''
        Return a copy with mirrored data (and mirrored landmarks).
        '''
        import numpy
        _A=numpy.array
        ret = DataRow() 
        ret.image=cv2.flip(self.image.copy(),1)
        # Now we mirror the landmarks and swap left and right
        width = ret.image.shape[0] 
        ret.leftEye = _A([width-self.rightEye[0], self.rightEye[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.rightEye = _A([width-self.leftEye[0], self.leftEye[1]])
        ret.middle = _A([width-self.middle[0], self.middle[1]])        
        ret.leftMouth = _A([width-self.rightMouth[0], self.rightMouth[1]]) # Toggle mouth positions and mirror x axis only
        ret.rightMouth = _A([width-self.leftMouth[0], self.leftMouth[1]])
        return ret

    @staticmethod
    def dummyDataRow(index):
        ''' Returns a dummy dataRow object to play with
        '''
        return DataRow('/Users/ishay/VanilaCNN/data/train/lfw_5590/Abbas_Kiarostami_0001.jpg',
                     leftEye=(106.75, 108.25),
                     rightEye=(143.75,108.75) ,
                     middle = (131.25, 127.25),
                     leftMouth = (106.25, 155.25),
                     rightMouth =(142.75,155.25)
                     )    
        
  
            
class Predictor:
    ROOT = getGitRepFolder() 
    
    def preprocess(self, resized, landmarks):
        ret = resized.astype('f4')
        ret -= self.mean
        ret /= (1.e-6+ self.std)
        return  ret, (landmarks/40.)-0.5
    
    def predict(self, resized):
        """
        @resized: image 40,40 already pre processed 
        """         
        self.net.blobs['data'].data[...] = cv2.split(resized)
        prediction = self.net.forward()['Dense2'][0]
        return prediction

    def predictReturnFeatureVectorAsWell(self, resized):
        """
        @resized: image 40,40 already pre processed 
        output: prediction, feature vector
        """         
        self.net.blobs['data'].data[...] = cv2.split(resized)
        prediction = self.net.forward()['Dense2'][0]
        featrueVector=self.net.blobs['ActivationAbs4'].data[0]
        return prediction, featrueVector

    def getFeatureVector(self, resized):
        self.net.blobs['data'].data[...] = cv2.split(resized)
        fvector = self.net.forward(start='Conv1', end='ActivationAbs4')['ActivationAbs4']
        return  fvector

        
    def __init__(self, protoTXTPath, weightsPath):
        import caffe
        caffe.set_mode_cpu()
        self.net = caffe.Net(protoTXTPath, weightsPath, caffe.TEST)
        self.mean = cv2.imread(os.path.join(Predictor.ROOT, 'trainMean.png')).astype('float')
        self.std  = cv2.imread(os.path.join(Predictor.ROOT,'trainSTD.png')).astype('float')


if __name__=='__main__':
    d=DataRow.dummyDataRow()
    d.show()

