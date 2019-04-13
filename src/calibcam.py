#!/usr/bin/env python
#python 2 and 3 compatibility
from __future__ import print_function

from common import *


def undistort(image, camera_matrix, dist_coefs, outfile):
        h, w = image[0].shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv.undistort(image, camera_matrix,
                           dist_coefs, None, newcameramtx)
        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

@static_vars(counter=0)
def calibrate(images):
    square_size = 3.0
    calibrate.counter += 1
    pattern_size = (6, 8)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    h, w = images[0].shape[:2]
    obj_points = []
    img_points = []
    debug_dir = './output/'

    def processImage(img, num):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img is None:
            print("Failed to load")
            return None
        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (
            img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            outfile = os.path.join(debug_dir, str(num) + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None
        print('           %s... OK' % str(num) + '_chess.png')
        return (corners.reshape(-1, 2), pattern_points)

#    chessboards = [processImage(img) for img in images]
    i = 0
    chessboards = []
    for img in images:
        i += 1
        chessboards.append(processImage(img, i))
    del i
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, (w, h), None, None)
    # undistort the image with the calibration
    print("RMS", rms)
    print('')
    xmlf = XmlFile("distortion_"+str(calibrate.counter)+".xml")
    xmlf.writeToXml('dist_matrix', camera_matrix)
    del xmlf
    xmlf = XmlFile("intrisics_"+str(calibrate.counter)+".xml")
    xmlf.writeToXml('matrix', camera_matrix)
    del xmlf
    print('Done')

def init():
    cam_number = int(
        input("Enter the webcam camera number as your system identifies it: "), 10)
    cap = cv.VideoCapture(cam_number)
    print("here")
    seconds = 1
    fps = cap.get(cv.CAP_PROP_FPS)  # Gets the frames per second
    print(fps)
    multiplier = fps * seconds
    images = []
    os.system("mkdir ./output")
    os.system("mkdir ./output/xmls")
    cv.namedWindow('Original')
    frameId = 0
    return cap, seconds, fps, multiplier, images, frameId


def main(cap, seconds, fps, multiplier, images, frameId):
    while True:
        # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        frameId += 1
        flag, image = cap.read()

        if ((frameId % multiplier) == 0):
            images.append(image)
            print("image collected")
        if(len(images) >= 5):
            calibrate(images)
            images.clear()
        if flag:
            # The frame is ready and already captured
            cv.imshow('Original', image)
        else:
            print("frame is not ready")
            # It is better 1to wait for a while for the next frame to be ready
            cv.waitKey(10)
        if cv.waitKey(25) == 27:
            cap.release()
            break

if __name__ == '__main__':
    main(*init())
    cv.destroyAllWindows()
