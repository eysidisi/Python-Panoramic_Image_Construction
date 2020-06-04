import glob
import os
import time
import cv2 as cv
import numpy as np
import Utils

if __name__ == "__main__":

    # file paths for input and output
    img_dir = r"C:\Users\AliIhsan\Desktop\Assignment_2\pano1"
    result_dir = r"C:\Users\AliIhsan\Desktop\Assignment_2\Results"

    # read all images and stack them in a list
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    img_list_1 = []
    for f1 in files:
        img = cv.imread(f1, 0)
        img_list_1.append(img)

    # temp image list for copying resultant images
    img_list_2 = []

    # an index to use while storing the images
    image_name_index = 0

    # keep pairing images until one image left in list
    while len(img_list_1) > 1:

        # image index in im_list_1
        img_index = 0

        while True:
            # if all images in list_1 are done, break
            if img_index == len(img_list_1):
                break

            # get next image from the list
            img_1 = img_list_1[img_index]
            # this copy is used for drawing the feature positions
            img_1_org = cv.copyTo(img_1, None)

            # if just one image is left in list append it and break
            if img_index + 1 == len(img_list_1):
                img_list_2.append(img_1)
                break

            # get next image from the list
            img_2 = img_list_1[img_index + 1]
            # this copy is used for drawing the feature positions
            img_2_org = cv.copyTo(img_2, None)

            # Initiate ORB detector
            orb = cv.ORB_create()

            # find the key points with ORB
            start = time.time()
            kp_1, des_1 = orb.detectAndCompute(img_1, None)
            kp_2, des_2 = orb.detectAndCompute(img_2, None)
            elapsed_time_fl = (time.time() - start)
            print("elapsed time to orb compute : " + str(elapsed_time_fl))

            # if key points couldn't be found: crop some part of the images, concatenate, append to the list and
            # continue
            if des_1 is None or des_2 is None:
                print("Cant find any key points in image. Cropping and concatenating images...")
                img_1_crop_ratio = 0.2
                img_2_crop_ratio = 0.2
                img_1 = img_1[:, 0:int(img_1.shape[1] * (1 - img_1_crop_ratio))]
                img_2 = img_2[:, int(img_2.shape[1] * img_2_crop_ratio):]
                concat_im = np.hstack((img_1, img_2))
                img_list_2.append(concat_im)
                img_index += 2
                continue

            # find desired num of good matches using ratio test
            desired_num_of_matches = 15
            start = time.time()
            try:
                good_matches_list, good_matches = Utils.find_matches(des_1, des_2, desired_num_of_matches)


            # if enough num of matches couldn't be found, crop some part of the images, concatenate, append to the list
            # and continue
            except:
                print("Cant find enough num of matches. Cropping and concatenating images...")
                img_1_crop_ratio = 0.2
                img_2_crop_ratio = 0.2
                img_1 = img_1[:, 0:int(img_1.shape[1] * (1 - img_1_crop_ratio))]
                img_2 = img_2[:, int(img_2.shape[1] * img_2_crop_ratio):]
                concat_im = np.hstack((img_1, img_2))
                img_list_2.append(concat_im)
                img_index += 2
                continue
            elapsed_time_fl = (time.time() - start)
            print("elapsed time to find matches: " + str(elapsed_time_fl))
            print("num of matches: " + str(len(good_matches_list)))

            # get positions of matches
            feature_ps_img_1 = np.float32([kp_1[m[0].queryIdx].pt for m in good_matches_list])
            feature_ps_img_2 = np.float32([kp_2[m[0].trainIdx].pt for m in good_matches_list])

            # get homography matrix
            start = time.time()
            distance_threshold = 10
            homography_mat = Utils.calculate_homography_matrix(feature_ps_img_1, feature_ps_img_2, distance_threshold)
            elapsed_time_fl = (time.time() - start)
            print("elapsed time to find homography mat: " + str(elapsed_time_fl))

            # position of match 1 in img_1
            x_pos = feature_ps_img_1[0][0].astype(int)
            y_pos = feature_ps_img_1[0][1].astype(int)

            # calculate corresponding points in img_2
            x_y_prime_mat = Utils.multiply_w_homography_mat(x_pos, y_pos, homography_mat)
            x_d_pos = (x_y_prime_mat[0] / x_y_prime_mat[2])[0].astype(int)
            y_d_pos = (x_y_prime_mat[1] / x_y_prime_mat[2])[0].astype(int)

            # cut unneeded parts
            img_1 = img_1[:, 0:x_pos]
            img_2 = img_2[:, x_d_pos:]

            # paste two image pair and append to list_2
            concat_im = np.hstack((img_1, img_2))
            img_list_2.append(concat_im)

            # draw only key points location
            img_1_w_key_points = \
                cv.drawKeypoints \
                    (img_1_org, kp_1, None, color=(0, 255, 0), flags=0)
            img_2_w_key_points = \
                cv.drawKeypoints \
                    (img_2_org, kp_2, None, color=(0, 255, 0), flags=0)
            matches_image = cv.drawMatches(img_1_org, kp_1, img_2_org, kp_2, good_matches, None, flags=2)

            # paths for writing
            img_1_w_key_points_file_path = result_dir + "\\" + str(image_name_index) + "_img_1_key_points.png"
            img_2_w_key_points_file_path = result_dir + "\\" + str(image_name_index) + "_img_2_key_points.png"
            matches_image_file_path = result_dir + "\\" + str(image_name_index) + "_img_1_img_2_matches.png"
            concat_image_file_path = result_dir + "\\" + str(image_name_index) + "_img_1_img_2_concat.png"

            # write results to disk
            cv.imwrite(img_1_w_key_points_file_path, img_1_w_key_points)
            cv.imwrite(img_2_w_key_points_file_path, img_2_w_key_points)
            cv.imwrite(matches_image_file_path, matches_image)
            cv.imwrite(concat_image_file_path, concat_im)

            img_index += 2
            image_name_index += 1

        # put found pairs in list_1 and clear list_2 to start over again
        img_list_1 = img_list_2.copy()
        img_list_2.clear()

    # resultant panorama image
    cv.imshow("im", img_list_1[0])
    panorama_im_path = result_dir + "\\" + "panorama.png"
    cv.imwrite(panorama_im_path, img_list_1[0])

    cv.waitKey()
    cv.destroyAllWindows()

    quit()
