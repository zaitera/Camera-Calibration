import os
import numpy as np
import cv2 as cv

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
