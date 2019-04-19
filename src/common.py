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
pointr1 = []
pointr2 = []
pointd1 = []
pointd2 = []

mode__ = str()

def init():
    print('''Hi, Usage:
        - after collecting at least 5 images, you can click the character c to initialize the calibration process for that dataset.
        - To get the undistort windows, the process of calibration needs to be run at least 5 times, clicking c each time.
        - make sure the c character was clicked, if the program recognized it it'll print a flag of starting the calibration.
    ''')
    cam_number = int(
        input("Enter the webcam camera number as your system identifies it: "), 10)
    cap = cv.VideoCapture(cam_number)
    seconds = float(input(
        "Enter the amount of time between the frames choosed for calibration in seconds (accepts float values): "))
    aux = input("To use solvePNP for extrinsics press (1) for normal method click anykey: ")
    if aux == "1":
        mode__ = "solvePNP"
        print("Setting mode to solvePNP")
        pass
    fps = cap.get(cv.CAP_PROP_FPS)  # Gets the frames per second
    print(str(fps)+" FPS")
    multiplier = fps * seconds
    images = deque(maxlen=5)
    os.system("mkdir ./output")
    os.system("mkdir ./output/xmls")
    print("deleting old xml files")
    os.system("rm ./output/xmls/*.xml")
    frameId = 0
    return cap, multiplier, images, frameId

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

def matricesPreparation():
    distortion_matrix = averageMatrixCaluclator("distortion")
    camera_matrix = averageMatrixCaluclator("intrinsics")
    extrinsics_matrix = averageMatrixCaluclator("extrinsics")
    writeXmlsAvgs(distortion_matrix, camera_matrix, extrinsics_matrix)
    writeXmlStds()
    return camera_matrix, extrinsics_matrix, distortion_matrix

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

def writeXmlStds():
    xmlf=XmlFile("stddistortion.xml")
    xmlf.writeToXml('matrix', stdMatrixCaluclator("distortion"))

    xmlf=XmlFile("stdintrinsics.xml")
    xmlf.writeToXml('matrix', stdMatrixCaluclator("intrinsics"))

    xmlf = XmlFile("stdextrinsics.xml")
    xmlf.writeToXml('matrix', stdMatrixCaluclator("extrinsics"))
    del xmlf


def writeXmlsAvgs(distortion_matrix, camera_matrix, extrinsics_matrix):
    xmlf = XmlFile("avgdistortion.xml")
    xmlf.writeToXml('matrix', distortion_matrix)

    xmlf = XmlFile("avgintrinsics.xml")
    xmlf.writeToXml('matrix', camera_matrix)

    xmlf = XmlFile("avgextrinsics.xml")
    xmlf.writeToXml('matrix', extrinsics_matrix)
    del xmlf

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
