import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.misc import imread, imsave, imresize
from scipy.io import loadmat
from scipy import ndimage
import math
from scipy.ndimage import gaussian_filter

"""
Spring 19 CS 2770 Assignment_1  
Implemented by Hao Zhao(haz92@pitt.edu) Upload 2/12/2019
"""


"""
Part I: Image Responses with Filters
"""
def part1(imagesList, leung_malik):
    """
    For each filter, generate a 2 * 4 subplots
    """
    for i in range(len(imagesList)):
        imagesList[i] = np.dot(imagesList[i], np.array([0.299, 0.587, 0.114]))
        imagesList[i] = cv2.resize(imagesList[i], [100, 100])

    if not os.path.exists("PartI_Subplots"):
        os.mkdir("PartI_Subplots")

    for i in range(48):
        f, ax = plt.subplots(2, 4)
        ax[0, 0].imshow(leung_malik[:, :, i])
        ax[0, 0].set_title("filter {}".format(i + 1))
        ax[0, 1].axis("off")
        ax[0, 2].imshow(ndimage.convolve(imagesList[0], leung_malik[:, :, i]))
        ax[0, 2].set_title(onlyFiles[0])
        ax[0, 3].imshow(ndimage.convolve(imagesList[1], leung_malik[:, :, i]))
        ax[0, 3].set_title(onlyFiles[1])
        ax[1, 0].imshow(ndimage.convolve(imagesList[2], leung_malik[:, :, i]))
        ax[1, 0].set_title(onlyFiles[2])
        ax[1, 1].imshow(ndimage.convolve(imagesList[3], leung_malik[:, :, i]))
        ax[1, 1].set_title(onlyFiles[3])
        ax[1, 2].imshow(ndimage.convolve(imagesList[4], leung_malik[:, :, i]))
        ax[1, 2].set_title(onlyFiles[4])
        ax[1, 3].imshow(ndimage.convolve(imagesList[5], leung_malik[:, :, i]))
        ax[1, 3].set_title(onlyFiles[5])
        filter_name = 'filter' + str(i + 1) + '.png'
        f.savefig(os.path.abspath(os.path.join("PartI_Subplots", filter_name)))
        plt.close(f)

"""
same_animal_similar: filter_1
different_animals_similar: filter_41
"""

"""
Part II: Image Responses with Filters
"""

def compute_texture_reprs(image, filter):
    # initialize the variable response
    # image_gray = np.dot(image, np.array([0.299, 0.587, 0.114]))
    image_gray = image
    img = cv2.resize(image_gray, (100, 100))
    response = np.ndarray(shape=(filter.shape[2], img.shape[0], img.shape[1]))
    # calculate the response
    for i in range(filter.shape[2]):
        new_img = ndimage.convolve(img, filter[:, :, i])
        response[i] = new_img

    # reshape the response to num_filter * 1
    texture_repr_concat = response.reshape(filter.shape[2] * img.shape[0] * img.shape[1])
    # calculate the texture_ repr_mean which is the mean of each response
    texture_repr_mean = []
    for i in range(filter.shape[2]):
        texture_repr_mean.append(np.mean(response[i]))

    return texture_repr_concat, texture_repr_mean

"""
Part III: Hybrid Images
"""

def part3(img1, img2):
    # resize the image to 512 * 512
    im1_resize = cv2.resize(img1, [512, 512])
    im2_resize = cv2.resize(img2, [512, 512])

    # convert color image to GrayScale
    im1_gray = cv2.cvtColor(im1_resize, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_resize, cv2.COLOR_BGR2GRAY)

    # use Gaussian_filter to blur images
    im1_blur = ndimage.gaussian_filter(im1_gray, sigma=1)
    im2_blur = ndimage.gaussian_filter(im2_gray, sigma=1)

    # subtract the blur part to get the detail part
    im2_detail = im2_gray - im2_blur

    # get hybrid by combining img1_blur and img2_detail
    hybrid = im1_blur + im2_detail
    cv2.imshow('Hybrid', hybrid)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.abspath(os.path.join("Part3_img", 'Hybrid' + '.png')), hybrid)

"""
Part IV: Feature Detection
"""
def extract_keypoints(img):
    # read the image(leopard2.jpg) and convert to the GrayScale
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img.astype("float64")
    # initialize the parameters
    # img_gray = img.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_gray = np.dot(img, np.array([0.299, 0.587, 0.114]))
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = 0.05
    window_size = 5
    height = img_gray.shape[0]
    width = img_gray.shape[1]

    # calculate the gradient of X and Y
    dy, dx = np.gradient(img_gray)

    # initialize with zeros a matrix R of the same size
    R = np.zeros((img_gray.shape[0], img_gray.shape[1]))
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy
    M = np.zeros((2, 2))
    cornerList = []
    offset = math.floor(window_size / 2)
    scores = []

    # Calculate the sum of squares
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            windowIxx = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1]
            M[0, 0] = windowIxx.sum()

            windowIxy = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1]
            M[0, 1] = windowIxy.sum()
            M[1, 0] = M[0, 1]

            windowIyy = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1]
            M[1, 1] = windowIyy.sum()

            # Find determinant and trace, then get response
            det = (M[1, 1] * M[0, 0]) - (M[0, 1] ** 2)
            trace = M[0, 0] + M[1, 1]
            R[i, j] = det - k * (trace ** 2)

    # the threshold setting in the OpenCV
    thresh = 0.01 * R.max()
    # img[R > 0.001 * R.max()] = (0, 0, 255)

    # the border does not have 8 neighbors, so get rid of four borders
    for i in range(1, img_gray.shape[0] - 1):
        for j in range(1, img_gray.shape[1] - 1):
            if R[i, j] >= R[i - 1:i + 1 + 1, j - 1:j + 1 + 1].max() and R[i, j] > thresh:
                scores.append(R[i, j])  # scores, i, j
                cornerList.append((i, j))

    # draw the circle via the number of scores
    for i in range(len(cornerList)):
        radius = math.floor(scores[i] / 50000000)
        m = cornerList[i][0]
        n = cornerList[i][1]
        cv2.circle(img, (n, m), radius, (0, 255, 0))
    # show the image
    cv2.imshow('image', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    name = 'part4' + '.png'
    cv2.imwrite(os.path.abspath(name), img)
    return cornerList, dx, dy

"""
Part V: Feature Description
"""
def compute_features(cornerList, dx, dy, image):
    # initialize the output of features(zeros matrix 339 * 8)
    features = np.zeros(shape=(len(cornerList), 8))
    # convert the image to Gray_Scale
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image.astype("float64")
    # for each keypoint we need to get its histogram
    for i in range(len(cornerList)):
        # initialize the array of histogram
        local = [0] * 8
        # get rid of the point which border is less than 5
        if cornerList[i][0] - 5 >= 0 and cornerList[i][0] + 5 <image_gray.shape[0] \
                and cornerList[i][1] - 5 >= 0 and cornerList[i][1] + 5 < image_gray.shape[1]:
            # the formula of the gradient magnitude
            grad_mag = np.sqrt((dx[cornerList[i][0]-5 : cornerList[i][0]+6, cornerList[i][1]-5: cornerList[i][1]+6]) ** 2\
                               + (dy[cornerList[i][0]-5 : cornerList[i][0] + 6, cornerList[i][1]-5 : cornerList[i][1]+6] ** 2))
            # the formula of the theta
            theta = np.zeros(shape=(11, 11))
            # if the dx is 0, we let theta become MAX_INT or MIN_INT
            for a in range(-5 , 6):
                for b in range(-5, 6):
                    if (dx[cornerList[i][0]+a, cornerList[i][1]+b] != 0):
                        theta[a + 5][b + 5] = (dy[cornerList[i][0]+a, cornerList[i][1]+b])\
                    /(dx[cornerList[i][0]+a, cornerList[i][1]+b] )
                    else:
                        theta[a + 5][b + 5] = dy[cornerList[i][0]+a, cornerList[i][1]+b] * 10000000

            # Pi
            P = math.pi
            # put the gradient magnitude to the histogram
            for a in range(grad_mag.shape[0]):
                for b in range(grad_mag.shape[1]):
                    if grad_mag[a][b] == 0:
                        continue
                    if math.atan(theta[a][b]) >= -P / 2 and math.atan(theta[a][b]) <= -P * 3 / 8:
                        local[0] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > -P * 3 / 8 and math.atan(theta[a][b]) <= -P / 4:
                        local[1] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > -P / 4 and math.atan(theta[a][b]) <= -P / 8:
                        local[2] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > -P / 8 and math.atan(theta[a][b]) <= 0:
                        local[3] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > 0 and math.atan(theta[a][b]) <= P / 8:
                        local[4] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > P / 8 and math.atan(theta[a][b]) <= P / 4:
                        local[5] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > P / 4 and math.atan(theta[a][b]) <= P * 3 / 8:
                        local[6] += grad_mag[a][b]
                    if math.atan(theta[a][b]) > P * 3 / 8 and math.atan(theta[a][b]) <= P / 2:
                        local[7] += grad_mag[a][b]

            # normalize -> clip -> normalize
            norm = [i / sum(local) for i in local]
            norm = np.clip(norm, 0, 0.2)
            norm = [i / sum(norm) for i in norm]
            # store all the information of features into the features matrix
            for j in range(len(norm)):
                features[i][j] = norm[j]
    return features

"""
Part VI: Image Description with SIFT Bag-of-Words
"""
def computeBOWRepr(features, means):
    # initialize the bow variable
    bow = [0] * means.shape[0]

    for i in range(features.shape[0]):
        dist = [0] * means.shape[0]
        for j in range(means.shape[0]):
            dist[j] = np.linalg.norm(features[i]- means[j])
        index = np.argmin(dist)
        bow[index] += 1
    bow_repr = [i / sum(bow) for i in bow]
    return bow_repr

# calculate the within_distance of images
def within_distance(img1, img2):
    x = loadmat('filters.mat')
    leung_malik = x['F']
    dist = []
    # calculate texture_repr_concat and texture_repr_mean
    texture_repr_concat1, texture_repr_mean1 = compute_texture_reprs(img1, leung_malik)
    texture_repr_concat2, texture_repr_mean2 = compute_texture_reprs(img2, leung_malik)
    dist.append(np.linalg.norm(texture_repr_concat1 - texture_repr_concat2))
    dist.append(np.linalg.norm(np.array(texture_repr_mean1) - np.array(texture_repr_mean2)))

    # calculate cornerList and scores
    # img1 = np.dot(img1, np.array([0.299, 0.587, 0.114]))
    # img2 = np.dot(img2, np.array([0.299, 0.587, 0.114]))
    cornerList1, dx1, dy1 = extract_keypoints(img1)
    cornerList2, dx2, dy2 = extract_keypoints(img2)

    # calculate features
    features1 = compute_features(cornerList1, dx1, dy1, img1)
    features2 = compute_features(cornerList2, dx2, dy2, img2)

    # calculate bow_repr
    bow_repr1 = computeBOWRepr(features1, means['means'])
    bow_repr2 = computeBOWRepr(features2, means['means'])
    dist.append(np.linalg.norm(np.array(bow_repr1) - np.array(bow_repr2)))
    return dist


# calculate the between_distance of images
def between_distance(img1, img2):
    x = loadmat('filters.mat')
    leung_malik = x['F']
    dist = []
    # calculate texture_repr_concat and texture_repr_mean
    texture_repr_concat1, texture_repr_mean1 = compute_texture_reprs(img1, leung_malik)
    texture_repr_concat2, texture_repr_mean2 = compute_texture_reprs(img2, leung_malik)
    dist.append(np.linalg.norm(texture_repr_concat1 - texture_repr_concat2))
    dist.append(np.linalg.norm(np.array(texture_repr_mean1) - np.array(texture_repr_mean2)))

    # calculate cornerList and scores
    cornerList1, dx1, dy1 = extract_keypoints(img1)
    cornerList2, dx2, dy2 = extract_keypoints(img2)

    # calculate features
    features1 = compute_features(cornerList1, dx1, dy1, img1)
    features2 = compute_features(cornerList2, dx2, dy2, img2)

    # calculate bow_repr
    bow_repr1 = computeBOWRepr(features1, means['means'])
    bow_repr2 = computeBOWRepr(features2, means['means'])
    dist.append(np.linalg.norm(np.array(bow_repr1) - np.array(bow_repr2)))
    return dist


def compare_description(imagesList):
    within = []
    for i in range(0, len(imagesList), 2):
        # imagesList[i] = imagesList[i].astype("float64")
        img1 = cv2.resize(imagesList[i], (100, 100))
        img2 = cv2.resize(imagesList[i + 1], (100, 100))
        dist = within_distance(img1, img2)
        within.append(dist)
    # within distance for three pairs images.
    re1 = 0; re2 = 0; re3 = 0
    for i in within:
        re1 += i[0]; re2 += i[1]; re3 += i[2]
    re1 /= 3; re2 /= 3; re3 /= 3
    print("Average Within_distance: texture_concat = {}, texture_mean = {}, bow_repr = {}".format(re1, re2, re3))

    between = []
    for i in range(len(imagesList)):
        for j in range(len(imagesList)):
            if j == i or (i % 2 == 0 and j == i + 1) or (i % 2 != 0 and j == i - 1):
                continue
            img1 = cv2.resize(imagesList[i], (100, 100))
            img2 = cv2.resize(imagesList[j], (100, 100))
            dist = between_distance(img1, img2)
            between.append(dist)
    bet_re1 = 0
    bet_re2 = 0
    bet_re3 = 0
    for i in between:
        bet_re1 += i[0]
        bet_re2 += i[1]
        bet_re3 += i[2]
    bet_re1 /= 24
    bet_re2 /= 24
    bet_re3 /= 24
    print("Average Between_distance: texture_concat = {}, texture_mean = {}, bow_repr = {}".format(bet_re1, bet_re2, bet_re3))
    print("Ratio: texture_concat = {}, texture_mean = {}, bow_repr = {}".format(re1 / bet_re1, re2 / bet_re2, re3 / bet_re3))

if __name__ == '__main__':
    # imagesPath is the path of the images
    imagesPath = os.path.abspath(os.path.join("pics"))

    # using a onlyFiles array to store the images
    onlyFiles = []
    images = os.listdir(imagesPath)
    images.sort()
    for i in images:
        if os.path.isfile(os.path.abspath(os.path.join(imagesPath, i))):
            onlyFiles.append(i)

    """      
    ImagesList is to store the matrix of images
    """
    imagesList = []
    for i in onlyFiles:
        img = cv2.imread(os.path.join(imagesPath, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagesList.append(img)

    # load the filer matrix
    x = loadmat('filters.mat')
    leung_malik = x['F']

    # x is the dict type, the last key is 'F' which store
    # the whole 48 filter (49 * 49)
    """
    PartI:
    """
    # part1(imagesList, leung_malik)
    """
    PartII:
    """
    # texture_repr_concat, texture_repr_mean = compute_texture_reprs(imagesList[0], leung_malik)
    """     
    PartIII: 
    """
    # part3_imagesPath = os.path.abspath(os.path.join("Part3_img"))
    # print(part3_imagesPath)
    # img1 = cv2.imread(os.path.join(part3_imagesPath, 'baby_happy.jpg'))
    # img2 = cv2.imread(os.path.join(part3_imagesPath, 'baby_weird.jpg'))
    # part3(img1, img2)
    """
    PartIV: Feature Detection
    """
    # scores = []
    # new_image = imread('./pics/panda2.jpg', mode="L")
    # cornerList, dx, dy = extract_keypoints(new_image)

    """
    PartV:Feature Description
    """
    # features = compute_features(cornerList, dx, dy, new_image)

    """
    PartVI: Image Description with SIFT Bag-of-Words
    """
    means = loadmat('means.mat')
    # bow_repr = computeBOWRepr(features, means['means'])

    """
    PartVII: Comparison of Image Descriptions
    """
    # read in the cardinal, leopard, and panda images, then resize them
    compare_description(imagesList)