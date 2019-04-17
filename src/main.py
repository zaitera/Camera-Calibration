#!/usr/bin/env python

from common import *

def undistortImage(image, camera_matrix, dist_coefs):
    h, w = image.shape[:2]
    dist_coefs = np.array(dist_coefs)
    newcameramtx, roi = NewtonRaphsonUndistort.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), 0)
    map1, map2 = cv.initUndistortRectifyMap(camera_matrix, dist_coefs, np.eye(3), newcameramtx,
                                            (w, h), cv.CV_32FC1)
    image_corr_mine = cv.remap(
        image, map1, map2, interpolation=cv.INTER_CUBIC)
    return image_corr_mine

@static_vars(counter=0)
def calibrate(images):
    square_size = 3.0
    calibrate.counter += 1
    pattern_size = (8, 6)
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
        print('%s... OK' % str(num) + '_chess.png written with pattern to /output/')
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
    print("Calculating extrinsics")

    rvec = np.mean(np.array(rvecs),axis = 0)
    tvec = np.mean(np.array(tvecs), axis=0)
    rotation_matrix, __ = cv.Rodrigues(rvec)
    extrinsics_matrix = np.concatenate(
        [rotation_matrix, tvec], 1)
    xmlf = XmlFile("distortion_"+str(calibrate.counter)+".xml")
    xmlf.writeToXml('matrix', dist_coefs)
    del xmlf
    xmlf = XmlFile("intrinsics_"+str(calibrate.counter)+".xml")
    xmlf.writeToXml('matrix', camera_matrix)
    del xmlf
    xmlf = XmlFile("extrinsics_"+str(calibrate.counter)+".xml")
    xmlf.writeToXml('matrix', extrinsics_matrix)
    del xmlf
    print('Done')
    return camera_matrix, dist_coefs

def main(cap, multiplier, images, frameId):
    calibrated = False
    while True:
        # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        frameId += 1
        flag, image = cap.read()
        if flag:
            cv.imshow('Original', image)
            if ((cv.waitKey(15) & 0xFF == ord('c')) and (len(images)>=5)):
                print(CGREEN+"Starting calibration"+CEND)
                camera_matrix, distortion_matrix = calibrate(list(images))
                if(calibrate.counter >= 5):
                    calibrate.counter = 0
                    calibrated = True
                    distortion_matrix = averageMatrixCaluclator("distortion")
                    camera_matrix = averageMatrixCaluclator("intrinsics")
                    writeXmlsAvgs(distortion_matrix, camera_matrix)
                    writeXmlStds()
                images.clear()
            if (calibrated and frameId % 15 == 0): #update undistort image every 500ms (because camera is 30 fps)
                undistored_image = undistortImage(image, camera_matrix, distortion_matrix)
                cv.imshow('undistored', undistored_image)
            if ((frameId % multiplier) == 0):
                images.append(image)
                print("image collected")
            # The frame is ready and already captured
        else:
            print("frame is not ready")
            # It is better 1to wait for a while for the next frame to be ready
            cv.waitKey(10)
        if cv.waitKey(15) == 27:
            cap.release()
            break

if __name__ == '__main__':
    main(*init())
    cv.destroyAllWindows()
