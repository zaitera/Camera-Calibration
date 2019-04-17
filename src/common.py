import os
import numpy as np
import cv2 as cv
from collections import deque
from calibrationUtils import NewtonRaphsonUndistort

CEND = '\33[0m'
CBOLD = '\33[1m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CBLUE = '\33[34m'

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def averageMatrixCaluclator(mat):
    #mat = distortion or intrinsics
    import glob
    path = "./output/xmls/"+mat+"_*.xml"
    i = 0
    for filename in glob.glob(path):
        i += 1
    xmlfile = XmlFile(mat+"_1"".xml")
    avg_mat = xmlfile.readFromXml('matrix')
    for j in range(2,i+1):
        filename = mat+"_"+str(j)+".xml"
        xmlfile = XmlFile(filename)
        avg_mat += xmlfile.readFromXml('matrix')
    avg_mat /= i
    del i,j
    return avg_mat


def stdMatrixCaluclator(mat):
    #mat = distortion or intrinsics
    import glob
    path = "./output/xmls/"+mat+"_*.xml"
    i = 0
    for filename in glob.glob(path):
        i += 1
    xmlfile = XmlFile(mat+"_1"".xml")
    all_matrix = xmlfile.readFromXml('matrix')

    for j in range(2, i+1):
        filename = mat+"_"+str(j)+".xml"
        xmlfile = XmlFile(filename)
        all_matrix = np.dstack((all_matrix, xmlfile.readFromXml('matrix')))
    del i, j
    return np.std(all_matrix, 2, ddof=1)

class XmlFile:
    Name = ""
    def __init__(self, name):
        if not(name.endswith('.xml')):
            print("invalid extension for xml file, check the name.")
        else:
            self.Name = name

    def writeToXml(self, label, value):
        outfile = './output/xmls/'
        outfile = os.path.join(outfile,self.Name)
        f = cv.FileStorage(outfile, flags=1)
        f.write(name=label, val=value)
        f.release()

    def readFromXml(self, label):
        outfile = './output/xmls/'
        outfile = os.path.join(outfile,self.Name)
        f = cv.FileStorage(outfile, flags=0)
        value = f.getNode(label).mat()
        f.release()
        return value
