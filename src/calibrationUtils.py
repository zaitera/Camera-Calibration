import cv2
import numpy as np
import os
from os import path as osp
__author__ = 'Yohai Devir, edited by Abdullah Zaiter'

MAX_NEWTON_ITERATIONS = 40
N_BISECTION_ITERATIONS = 20

INITIAL_STEP_SIZE_PIX = 250
MIN_STEP_SIZE_PIX = 0.1

class NewtonRaphsonUndistort:
    def __init__(self):
        pass

    # region Internals - Python implementations of OpenCV functions
    @staticmethod
    def icv_get_rectangles(camera_matrix, dist_coeffs, new_camera_matrix, img_size):
        """
        Python implementation of OpenCV's internal function icvGetRectangles with a bugfix.
        In general, it finds the mapping of an distorted image of img_size to the undisrted image space.
        :param camera_matrix: distorted image camera matrix
        :param dist_coeffs: distortion parameters
        :param new_camera_matrix: undistorted image's camera matrix, or None if it is not known
        :param img_size: following openCV conventions, img_size is X/Y
        :return: outer - bounding rectangle of all distorted image pixels,
                 inner - maximal rectengle that contain only pixels from the distorted image (without any padded pixels
        """
        n = 9
        pts = np.zeros((n * n, 2))
        pt_idx = 0
        for y in range(n):
            for x in range(n):
                pts[pt_idx, 0] = float(x) * (img_size[0] - 1) / (n - 1)  # following openCV conventions, img_size is X/Y
                pts[pt_idx, 1] = float(y) * (img_size[1] - 1) / (n - 1)
                pt_idx += 1

        res_pts, estimation_errors = \
            NewtonRaphsonUndistort.cv_undistort_points(pts, camera_matrix, dist_coeffs, new_camera_matrix)

        float_max = float(1e10)

        i_x0 = -float_max
        i_x1 = float_max
        i_y0 = -float_max
        i_y1 = float_max
        o_x0 = float_max
        o_x1 = -float_max
        o_y0 = float_max
        o_y1 = -float_max

        #  find the inscribed rectangle.
        # the code will likely not work with extreme rotation matrices (R) (>45%)

        pt_idx = 0
        for y in range(n):
            for x in range(n):
                p = res_pts[pt_idx]
                pt_idx += 1
                o_x0 = min(o_x0, p[0])
                o_x1 = max(o_x1, p[0])
                o_y0 = min(o_y0, p[1])
                o_y1 = max(o_y1, p[1])

                if x == 0:
                    i_x0 = max(i_x0, p[0])
                if x == n - 1:
                    i_x1 = min(i_x1, p[0])
                if y == 0:
                    i_y0 = max(i_y0, p[1])
                if y == n - 1:
                    i_y1 = min(i_y1, p[1])

        inner = {'x': i_x0, 'y': i_y0, 'width': i_x1 - i_x0, 'height': i_y1 - i_y0}
        outer = {'x': o_x0, 'y': o_y0, 'width': o_x1 - o_x0, 'height': o_y1 - o_y0}

        return inner, outer

    @staticmethod
    def cv_undistort_points(pts_distorted_pix, camera_matrix, dist_coeffs, new_camera_matrix):
        """
        Python implementation of OpenCV's cvUndistortPoints with a bugfix.
        cv_undistort_points takes a list of pixel location in a distorted image and returns their corresponding
        locations in an undistorted image
        For details: https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html
        :param pts_distorted_pix:
        :param camera_matrix:
        :param dist_coeffs:
        :type new_camera_matrix: np.ndarray | NoneType
        :return:
        """

        if new_camera_matrix is None:
            new_camera_matrix = np.eye(3)

        dist_coeffs = dist_coeffs.ravel()

        assert len(dist_coeffs) <= 5

        pts_undistorted_pix = np.zeros_like(pts_distorted_pix)

        dist_coeffs = np.hstack((dist_coeffs, np.zeros(14 - len(dist_coeffs))))
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1 = dist_coeffs

        if tilt_param0 != 0 or tilt_param1 != 0:
            raise NotImplementedError("computeTiltProjectionMatrix was not implemented here in python")

        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        ifx = 1. / fx
        ify = 1. / fy
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        pixel_to_mm = 0.5 * (fx + fy)
        initial_step_size_mm = INITIAL_STEP_SIZE_PIX / pixel_to_mm
        min_step_size_mm = MIN_STEP_SIZE_PIX / pixel_to_mm

        n = len(pts_distorted_pix)
        estimation_errors = np.ones(n) * -1
        for i in range(n):
            x_dist_pix = pts_distorted_pix[i, 0]
            y_dist_pix = pts_distorted_pix[i, 1]

            x_dist_mm = (x_dist_pix - cx) * ifx
            y_dist_mm = (y_dist_pix - cy) * ify
            loc_dist_target = np.array([x_dist_mm, y_dist_mm])

            loc_orig, final_error = NewtonRaphsonUndistort.undistort_single_pixel(
                loc_dist_target, dist_coeffs, initial_step_size_mm=initial_step_size_mm,
                min_step_size_mm=min_step_size_mm)
            estimation_errors[i] = final_error

            x_mm, y_mm = loc_orig

            xx_pix = new_camera_matrix[0, 0] * x_mm + new_camera_matrix[0, 1] * y_mm + new_camera_matrix[0, 2]
            yy_pix = new_camera_matrix[1, 0] * x_mm + new_camera_matrix[1, 1] * y_mm + new_camera_matrix[1, 2]
            ww_pix = 1. / (new_camera_matrix[2, 0] * x_mm + new_camera_matrix[2, 1] * y_mm + new_camera_matrix[2, 2])
            x_final = xx_pix * ww_pix
            y_final = yy_pix * ww_pix

            pts_undistorted_pix[i][0] = float(x_final)
            pts_undistorted_pix[i][1] = float(y_final)

        return pts_undistorted_pix, estimation_errors

    # endregion

    # region Internals - better implementation of single pixel undistortion
    @staticmethod
    def _distort_pixel_and_calc_error(loc_orig, loc_dist_target, k1, k2, k3, p1, p2):
        x_orig, y_orig = loc_orig[0], loc_orig[1]
        r2 = x_orig * x_orig + y_orig * y_orig
        r4 = r2 ** 2
        r6 = r2 ** 3
        a1 = 2 * x_orig * y_orig
        a2 = r2 + 2 * x_orig * x_orig
        a3 = r2 + 2 * y_orig * y_orig
        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        delta_x = p1 * a1 + p2 * a2
        delta_y = p1 * a3 + p2 * a1

        x_dist = x_orig * cdist + delta_x
        y_dist = y_orig * cdist + delta_y
        loc_distorted = np.array([x_dist, y_dist])

        error_xy = loc_distorted - loc_dist_target
        return loc_distorted, error_xy

    @staticmethod
    def _error_jacobian(loc_orig, k1, k2, k3, p1, p2):
        """
        Calculate the Jacobian of distorted location of loc_orig minus desired distorted location.
        :param loc_orig:
        :param k1:
        :param k2:
        :param k3:
        :param p1:
        :param p2:
        :return:
        """
        x_orig, y_orig = loc_orig[0], loc_orig[1]
        r2 = x_orig * x_orig + y_orig * y_orig
        r4 = r2 ** 2
        r6 = r2 ** 3

        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6

        common = (k1 + 2 * k2 * r2 + 3 * k3 * r4)
        d_error_x_dx = (cdist + 2 * p1 * y_orig + 6 * p2 *
                        x_orig + 2 * x_orig * x_orig * common)
        d_error_y_dy = (cdist + 6 * p1 * y_orig + 2 * p2 *
                        x_orig + 2 * y_orig * y_orig * common)
        d_error_x_dy = 2 * (p1 * x_orig + p2 * y_orig +
                            x_orig * y_orig * common)
        d_error_y_dx = 2 * (p1 * x_orig + p2 * y_orig +
                            y_orig * x_orig * common)

        jacobian = np.array([[d_error_x_dx, d_error_x_dy],
                             [d_error_y_dx, d_error_y_dy]])
        return jacobian

    @staticmethod
    def undistort_single_pixel(loc_dist_target, dist_coeffs, initial_step_size_mm=0.05, min_step_size_mm=1e-8):
        """
        Given a pixel location in a distorted image, find its location in the undistorted image.
        As the distortion function is hard to invert, this implementation iteratively searches for a location in the
        original image that is mapped by the distortion function to the desired location.
        OpenCV implementation (in cvUndistortPoints) seems to assume that such a location exists and that the distortion
        function is monotonous.
        Both assumptions may not hold as the distortion parameters calculation does not force monotonicity.
        In this implementation, I perform Newton Raphson iterations with binary search for a decrease in the error.
        :param loc_dist_target: pixel location in the distorted image
        :param dist_coeffs: distortion paramets as in openCV
        :param initial_step_size_mm:
        :param min_step_size_mm:
        :return:
        """

        dist_coeffs = np.hstack((dist_coeffs, np.zeros(14 - len(dist_coeffs))))
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1 = dist_coeffs

        assert not np.any([k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1]), \
            "NewtonRaphsonUndistort was given distortion params of more than 5 variables, which is unsuipported"

        loc_orig = loc_dist_target  # initial guess is that there is no distortion
        step_size = initial_step_size_mm
        final_error = np.inf
        for ii in range(MAX_NEWTON_ITERATIONS):
            # Performing NewtonRaphson iterations on f(x,y) = distort(x,y) - target_location(x,y)
            # Single variable explanation: https://en.wikipedia.org/wiki/Newton%27s_method#Description
            # Multivariate version explanation: http://fourier.eng.hmc.edu/e161/lectures/ica/node13.html
            loc_dist, err_vec = \
                NewtonRaphsonUndistort._distort_pixel_and_calc_error(loc_orig, loc_dist_target, k1, k2, k3, p1, p2)
            l2_error_before = np.linalg.norm(err_vec)

            jacobian = NewtonRaphsonUndistort._error_jacobian(loc_orig, k1, k2, k3, p1, p2)

            current_movement = -1 * np.linalg.inv(jacobian).dot(err_vec.T)
            current_movement_size = np.linalg.norm(current_movement)

            # when the derivatives near the solution are small, a single step may take us far away from the solution.
            # In order to avoid this, I enforce that the error must decrease in every step.
            should_break = False
            next_loc_orig = l2_error_after = None
            for step_size_iter in range(N_BISECTION_ITERATIONS):
                if step_size_iter == N_BISECTION_ITERATIONS - 1:
                    should_break = True
                    break
                next_loc_orig = (loc_orig.T + step_size * current_movement / current_movement_size).T
                _, error_xy_after = NewtonRaphsonUndistort._distort_pixel_and_calc_error(
                    next_loc_orig, loc_dist_target, k1, k2, k3, p1, p2)

                l2_error_after = np.linalg.norm(error_xy_after)
                if l2_error_after < l2_error_before:
                    final_error = l2_error_after
                    break
                step_size *= 0.5
            if should_break:
                break

            assert next_loc_orig is not None and l2_error_after is not None
            loc_orig = next_loc_orig

            if final_error < min_step_size_mm:
                # in this case we stop due to convergence
                final_error = 0
                break

            if step_size < min_step_size_mm:
                break

        return loc_orig, final_error

    @staticmethod
    def getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, img_size, alpha):
        """
        Python implementation of OpenCV's cvGetOptimalNewCameraMatrix with a bugfix.
        For details: docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        :param camera_matrix:
        :param dist_coeffs:
        :param img_size:  following openCV conventions, img_size is X/Y
        :param alpha:
        :return:
        """

        # Get inscribed and circumscribed rectangles in normalized
        # (independent of camera matrix) coordinates
        inner, outer = NewtonRaphsonUndistort.icv_get_rectangles(camera_matrix, dist_coeffs, None, img_size)

        # Projection mapping inner rectangle to viewport
        fx0 = (img_size[0]) / inner['width']  # following openCV conventions, img_size is X/Y
        fy0 = (img_size[1]) / inner['height']
        cx0 = -fx0 * inner['x']
        cy0 = -fy0 * inner['y']

        # Projection mapping outer rectangle to viewport
        fx1 = (img_size[0]) / outer['width']  # following openCV conventions, img_size is X/Y
        fy1 = (img_size[1]) / outer['height']
        cx1 = -fx1 * outer['x']
        cy1 = -fy1 * outer['y']

        # Interpolate between the two optimal projections
        new_camera_matrix = np.zeros((3, 3))
        new_camera_matrix[0][0] = fx0 * (1 - alpha) + fx1 * alpha
        new_camera_matrix[1][1] = fy0 * (1 - alpha) + fy1 * alpha
        new_camera_matrix[0][2] = cx0 * (1 - alpha) + cx1 * alpha
        new_camera_matrix[1][2] = cy0 * (1 - alpha) + cy1 * alpha
        new_camera_matrix[2][2] = 1

        return new_camera_matrix, img_size
