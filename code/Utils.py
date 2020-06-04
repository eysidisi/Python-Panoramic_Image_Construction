import cv2 as cv
import numpy as np
import random
from numpy.linalg import inv
from datetime import datetime


def calc_b_mat():
    """Used for calculation of homography matrix"""
    b = []
    for i in range(8):
        b.append([0])
    b.append([1])
    return b


def multiply_w_homography_mat(x, y, h):
    """Calculates corresponding position points for given position points using homography matrix"""
    pos_mat = [[x], [y], [1]]
    pos_mat_prime = np.dot(h, pos_mat)
    return pos_mat_prime


def find_matches(des_1, des_2, num_of_min_desired_matches):
    """This function tries to find desired num of matches and raises and error if it can not find"""
    bf = cv.BFMatcher()
    matches_knn = bf.knnMatch(des_1, des_2, k=2)

    # try to calculate good matches with ratios test
    # ratio_val starts from a low value and increases
    ratio_val = 0.2
    good_matches = []
    while len(good_matches) < num_of_min_desired_matches and ratio_val < 0.8:
        ratio_val += 0.05
        good_matches = []
        for m, n in matches_knn:
            if m.distance < ratio_val * n.distance:
                good_matches.append(m)

    if len(good_matches) < num_of_min_desired_matches:
        raise NameError("Can't find enough of features")

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    good_matches_list = []
    for match in good_matches:
        good_matches_list.append([match])

    return good_matches_list, good_matches


def calc_distances(feature_ps_im_1, feature_ps_im_2, homography_mat):
    """calculates distances between two feature point sets"""
    distances = []
    for i in range(len(feature_ps_im_1)):
        x = feature_ps_im_1[i][0].astype(int)
        y = feature_ps_im_1[i][1].astype(int)

        x_y_prime_mat = multiply_w_homography_mat(x, y, homography_mat)
        if x_y_prime_mat[2] == 0:
            x_y_prime_mat[2] = 0.001
        x_d = (x_y_prime_mat[0] / x_y_prime_mat[2])[0].astype(int)
        y_d = (x_y_prime_mat[1] / x_y_prime_mat[2])[0].astype(int)

        x_2 = feature_ps_im_2[i][0].astype(int)
        y_2 = feature_ps_im_2[i][1].astype(int)

        try:
            dist = np.math.sqrt((x_2 - x_d) ** 2 + (y_2 - y_d) ** 2)
            distances.append(dist)
        except:
            print("error while calculating the dist")

    return distances


def calculate_homography_matrix(feature_ps_im_1, feature_ps_im_2, distance_threshold):
    """  calculates homography matrix using AH=b
     for more information https://math.stackexchange.com/q/2619023"""

    # number of samples to build the model
    n = 4

    # number of iterations
    k = 100

    num_of_points = len(feature_ps_im_1)
    best_homography_mat = None
    max_num_of_inliers = 0
    # index of iteration
    i = 0

    while i < k:

        # Pick n random points
        random.seed(datetime.now())
        random_numbers = random.sample(range(0, num_of_points), n)

        # create A matrix
        A_mat = []
        for num in random_numbers:
            x = int(feature_ps_im_1[num][0])
            y = int(feature_ps_im_1[num][1])
            x_d = int(feature_ps_im_2[num][0])
            y_d = int(feature_ps_im_2[num][1])
            A_mat.append([0, 0, 0, -x, -y, -1, x * y_d, y_d * y, y_d])
            A_mat.append([x, y, 1, 0, 0, 0, -x * x_d, -x_d * y, -x_d])
        A_mat.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        A_mat = np.asarray(A_mat).reshape(9, 9)

        if np.linalg.det(A_mat) == 0:
            continue

        # calculate homography matrix
        b_mat = calc_b_mat()
        A_inv = inv(A_mat)
        homography_mat = np.dot(A_inv, b_mat)
        homography_mat = np.reshape(homography_mat, (3, 3))

        distances = calc_distances(feature_ps_im_1, feature_ps_im_2, homography_mat)

        # calculate num of inliers
        num_of_inliers = 0
        for distance in distances:
            if distance < distance_threshold:
                num_of_inliers += 1

        # update best homography matrix
        if num_of_inliers > max_num_of_inliers:
            max_num_of_inliers = num_of_inliers
            best_homography_mat = homography_mat
        i += 1
    return best_homography_mat
