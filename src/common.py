import os
import numpy as np
import cv2 as cv

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

class XmlFile:
    Name = ""
    def __init__(self, name):
        if not(name.endswith('.xml')):
            print("invalid extension for xml file, check the name.")
        else:
            self.Name = name

    def writeToXml(self, label, value):
        f = cv.FileStorage(self.Name, flags=1)
        f.write(name=label, val=value)
        f.release()

    def readFromXml(self, label):
        f = cv.FileStorage(self.Name, flags=0)
        value = f.getNode(label).mat()
        f.release()
        return value
