from tabnanny import check
import cv2, os, csv, time, sys
from cv2 import resize
from cv2 import boundingRect
# sys.path.append("ocr/src")

import numpy as np
# import keras_ocr
import matplotlib.pyplot as plt
from spellchecker import spellchecker
import pandas as pd
from PIL import Image, ImageEnhance
from .main import main
#############START##################
from scipy.spatial import distance
import math
from termcolor import colored
import ocr.blue as blue
import shutil
#############END####################


def clear():
    for i in os.listdir('ocr/resize'):
        if not i.endswith('png') and not i.endswith('jpg'):
            for j in os.listdir(f'ocr/resize/{i}'):
                os.remove(f"ocr/resize/{i}/{j}")
        else:
            os.remove(f'ocr/resize/{i}')
    for k in os.listdir("ocr/ROI"):
        if k.endswith('.png') or k.endswith('.jpg'):
            os.remove(f'ocr/ROI/{k}')
            print("cleared")


clear()


def rgb(img):

    # copying original image from media directory to ocr/outputs
    shutil.copy(img, "ocr/outputs/basic_input.jpg")

    # reading original image and converting to hsv
    image = cv2.imread(img)
    bhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # computing file size to apply different settings for different size images
    file_size = round(os.path.getsize(img) / 10**6)

    if file_size > 4:  # setting for images having file_size > 3 MB

        # noise mask 1
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 5])
        noise_mask = cv2.inRange(bhsv, lower, upper)

        # noise mask 2
        lower = np.array([116, 0, 0])
        upper = np.array([144, 54, 75])
        noise_mask_2 = cv2.inRange(bhsv, lower, upper)

        # noise mask 3
        lower = np.array([109, 35, 0])
        upper = np.array([124, 255, 255])
        noise_mask_3 = cv2.inRange(bhsv, lower, upper)

        # extracting blue color using hsv ranges
        lower = np.array([99, 26, 0])
        upper = np.array([127, 255, 255])
        bmask = cv2.inRange(bhsv, lower, upper)

        # removing noise from extracted content by applying noise masks
        bmask[noise_mask == 255] = 0
        bmask[noise_mask_2 == 255] = 0
        bmask[noise_mask_3 == 0] = 0

        # removing remaining noise having area < 50
        cnts = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50:
                cv2.drawContours(bmask, [c], 0, 0, -1)

    else:  # setting for images having file_size <= 3 MB

        lower = np.array([106, 60, 0])
        upper = np.array([144, 255, 255])
        bmask = cv2.inRange(bhsv, lower, upper)
    
    # _, ib = cv2.threshold(bmask, 50, 200, cv2.THRESH_BINARY_INV)
    
    ib = 255 - bmask  # inverting mask to have black text with white background

    # saving image to ocr/outputs director and printing success message
    cv2.imwrite("ocr/outputs/ib.jpg", ib)
    print("Blue Image Generated")

    # print(j)
    ### finding red color only
    rhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # # rmask = cv2.inRange(rhsv, np.array([161, 155, 84]), np.array([179, 255, 255]))
    # # rmask = cv2.inRange(rhsv,np.array([0,20 ,60]),np.array([60,180,255]))
    # rmask = cv2.inRange(rhsv,np.array([170, 120, 70]),np.array([180, 255, 255]))
    # # rmask = cv2.inRange(rhsv,np.array([160, 150, 80]),np.array([170, 255, 255]))
    e = cv2.inRange(rhsv, np.array([170, 50, 50]),
                    np.array([180, 155, 150]))  ##Qualitive for newRed
    # f = cv2.inRange(rhsv,np.array([175,50,20]),np.array([180,255,255]))
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(rhsv, lower1, upper1)
    upper_mask = cv2.inRange(rhsv, lower2, upper2)

    full_mask = lower_mask + upper_mask + e
    rmask = full_mask

    # gmask = cv2.inRange(rhsv, np.array([40, 20, 50]), np.array([100, 100, 180]))    ## for pdf images
    gmask = cv2.inRange(rhsv, np.array([40, 60, 25]),
                        np.array([100, 255, 255]))  ## for normal image
    # gmask = cv2.inRange(rhsv,np.array([65,60,60]), np.array([80, 255,255]))

    _, ir = cv2.threshold(rmask, 100, 220, cv2.THRESH_BINARY_INV)
    # ir = cv2.erode(ir,kernel,iterations=1)
    cv2.imwrite("ocr/outputs/ir.jpg", ir)

    _, ig = cv2.threshold(gmask, 20, 250, cv2.THRESH_BINARY_INV)
    # ig = cv2.erode(ig, kernel, iterations=1)
    cv2.imwrite("ocr/outputs/ig.jpg", ig)

    ### For black
    # black_mask = cv2.inRange(bhsv, np.array([0, 0, 0]), np.array([150, 30, 150]))
    black_mask = cv2.inRange(bhsv, np.array([0, 0, 0]),
                             np.array([100, 185, 60]))
    res_bl = cv2.bitwise_and(image, image, mask=black_mask)
    _, iblack = cv2.threshold(res_bl, 5, 250,
                              cv2.THRESH_BINARY_INV)  ## inverted green
    cv2.imwrite("ocr/outputs/bl_text.jpg", iblack)


def extractROI(image):
    extimg = cv2.imread(image)
    gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    ##############START##################
    textColor = " "
    if image.endswith("ib.jpg"):
        textColor = "Blue"
    if image.endswith("ir.jpg"):
        textColor = "Red"
    if image.endswith("ig.jpg"):
        textColor = "Green"
    if image.endswith("bl.jpg"):
        textColor = "Black"
    ##############END####################

    dilate = cv2.dilate(thresh, kernel, iterations=6)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    ROI_number = 0
    '''
    textColor = " "
    if image.endswith("ib.jpg"):
        textColor = "Blue"
    if image.endswith("ir.jpg"):
        textColor = "Red"
    if image.endswith("ig.jpg"):
        textColor = "Green"
    if image.endswith("bl.jpg"):
        textColor = "Black"
    
    df = pd.DataFrame(
        {"ImageName": [],
         "y": [],
         "y+h": [],
         "x": [],
         "x+w": [],
         "TextColor": []
         },dtype=int
    )
    '''
    ################START################
    df = pd.DataFrame(
        {
            "ImageName": [],
            "y": [],
            "y+h": [],
            "x": [],
            "x+w": [],
            "TextColor": [],
            "dot_coordinate_x": [],
            "dot_coordinate_y": []
        },
        dtype=int)
    ################END##################
    for c in cnts:
        area = cv2.contourArea(c)
        const = 5000
        if textColor == "Red":
            const = 500
        elif textColor == "Blue":
            '''
            const = 5000
            '''
            ##########START##########
            const = 3000
            ##########END############
        if area > const:
            x, y, w, h = cv2.boundingRect(c)
            left = y, y + h
            right = x, x + w
            '''
            # pair = [left,right]
            ## creating a dataframe
            df2 = pd.DataFrame(
                {   "ImageName":["ROI_{}.jpg".format(ROI_number)],
                    "y":[y],
                    "y+h":[y+h],
                    "x":[x],
                    "x+w":[x+w],
                    "TextColor":[textColor]
                }
            )

            df = df.append(df2)
            '''
            ROI = extimg[left[0]:left[1], right[0]:right[1]]
            ###################START###########################
            # find dots' coordinates
            # Load image, grayscale, Otsu's threshold
            image = ROI.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            final_x, final_y = 0, 0
            for c in cnts:
                x1, y1, w1, h1 = cv2.boundingRect(c)
                area = w1 * h1
                area1 = cv2.contourArea(c)
                H, W = cv2.imread('ocr/outputs/basic_input.jpg').shape[:2]
                # filtering dots
                if x1 < 50 and area > (48 / 4032 * W) and area < (
                        650 / 4032 * W) and area1 != 0 and area / area1 <= 2:
                    # if x1 < 50 and area > 48 and area < 650 and area != 0 and area/area1 <= 2:
                    center_x = x1 + w1 / 2
                    center_y = y1 + h1 / 2
                    final_x = center_x + right[0]
                    final_y = center_y + left[0]

            # pair = [left,right]
            # creating a dataframe
            df2 = pd.DataFrame({
                "ImageName": ["ROI_{}.jpg".format(ROI_number)],
                "y": [y],
                "y+h": [y + h],
                "x": [x],
                "x+w": [x + w],
                "TextColor": [textColor],
                "dot_coordinate_x": [final_x],
                "dot_coordinate_y": [final_y]
            })

            df = df.append(df2)
            # df.loc[len(df.index)] = ["ROI_{}.jpg".format(ROI_number),
            #                          y, y+h, x, x+w, textColor, final_x, final_y]
            ###############END#######################
            cv2.imwrite('ocr/ROI/ROI_{}.png'.format(ROI_number), ROI)
            ROI_number += 1
            print("CONST :: ", const)
    return df


def scale_img():
    files = []
    for i in os.listdir("ocr/ROI"):
        if i.endswith('png'):
            files.append(i)
    '''
    files.sort()
    '''

    ################START##############
    def number(f):
        num = int(f.split('_')[1].split('.')[0])
        return num

    files.sort(key=number)
    ################END################

    # print(files)
    c = 0
    for j in files:
        im = cv2.imread("ocr/ROI/" + str(j), -1)

        resize_image = cv2.resize(im, (300, 100))  ### 275 -- 110

        cv2.imwrite("ocr/resize/ROI_{}.png".format(c), resize_image)
        # print(j,os.path.exists("ROI/" + str(j)))
        c += 1


def dilate_erode(col):
    ## DILATION & EROSION
    files = []
    for j in os.listdir("ocr/resize"):
        if j.endswith("png"):
            files.append(j)
    files.sort()
    # print("fnames : ", files)

    for i in files:
        img_ = cv2.imread('ocr/resize/' + str(i), -1)
        #         print(img_)
        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((3, 3), np.uint8)
        kernel_ = np.ones((5, 5), np.uint8)
        #         plt.figure(figsize=(9,9))
        #         plt.imshow(img_)
        img_erosion = cv2.erode(img_, kernel, iterations=2)
        img_er = cv2.erode(img_, kernel, iterations=1)
        img_dilation = cv2.dilate(img_, kernel_, iterations=1)
        er2 = cv2.erode(img_dilation, kernel, iterations=1)

        kernelsh = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp1 = cv2.filter2D(img_erosion, -1, kernelsh)

        di2 = cv2.dilate(img_erosion, kernel_, iterations=1)
        sharp2 = cv2.filter2D(di2, -1, kernelsh)

        gausBlur = cv2.GaussianBlur(img_, (5, 5), 0)
        medBlur = cv2.medianBlur(img_, 3)
        bilFilter = cv2.bilateralFilter(img_, 9, 75, 75)

        bi_ = cv2.bilateralFilter(img_erosion, 9, 75, 75)
        bi_di_ = cv2.bilateralFilter(di2, 9, 75, 75)

        sharp3 = cv2.filter2D(gausBlur, -1, kernelsh)
        sharp4 = cv2.filter2D(bi_, -1, kernelsh)
        sharp5 = cv2.filter2D(medBlur, -1, kernelsh)
        sharp6 = cv2.filter2D(bi_di_, -1, kernelsh)

        if col == "blue":
            image = cv2.imread('ocr/resize/' + str(i), -1)

            image = cv2.resize(image, (600, 150))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Morph open to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(thresh,
                                       cv2.MORPH_OPEN,
                                       kernel,
                                       iterations=1)

            # Find contours and remove small noise
            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 10:
                    cv2.drawContours(opening, [c], -1, 0, -1)

            # Invert and apply slight Gaussian blur
            result = 255 - opening

            img_dilation = cv2.dilate(result, kernel, iterations=1)

            img_dilation = cv2.erode(img_dilation, kernel, iterations=2)

            result = img_dilation

            result = cv2.GaussianBlur(result, (3, 3), 0)

            cv2.imwrite('ocr/resize/morph/morph_{}.jpg'.format(i[:len(i) - 4]),
                        result)

        elif col == "red":
            image = cv2.imread('ocr/ROI/' + str(i), -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            image = cv2.erode(image, kernel, iterations=2)
            # image = cv2.dilate(image, kernel, iterations=1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh,
                                       cv2.MORPH_OPEN,
                                       kernel,
                                       iterations=1)
            # Find contours and remove small noise
            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 10:
                    cv2.drawContours(opening, [c], -1, 0, -1)
            # Invert and apply slight Gaussian blur
            result = 255 - opening
            img_dilation = cv2.dilate(result, kernel, iterations=1)
            result = img_dilation
            result = cv2.GaussianBlur(result, (5, 5), 0)
            cv2.imwrite('ocr/resize/red/red_{}.jpg'.format(i[:len(i) - 4]),
                        result)

        # cv2.imwrite('Input', img)
        cv2.imwrite('ocr/resize/erode/Erode_{}.jpg'.format(i[:len(i) - 4]),
                    img_erosion)
        cv2.imwrite('ocr/resize/erode1/Erode_{}.jpg'.format(i[:len(i) - 4]),
                    img_er)
        cv2.imwrite('ocr/resize/dilate/Dilate_{}.jpg'.format(i[:len(i) - 4]),
                    img_dilation)
        cv2.imwrite('ocr/resize/gauss/gauss_{}.jpg'.format(i[:len(i) - 4]),
                    gausBlur)
        cv2.imwrite('ocr/resize/median/median_{}.jpg'.format(i[:len(i) - 4]),
                    medBlur)
        cv2.imwrite('ocr/resize/bifilter/bi_{}.jpg'.format(i[:len(i) - 4]),
                    bilFilter)
        cv2.imwrite('ocr/resize/di_/di_{}.jpg'.format(i[:len(i) - 4]), di2)
        cv2.imwrite('ocr/resize/er_/er_{}.jpg'.format(i[:len(i) - 4]), er2)
        cv2.imwrite('ocr/resize/er_bi_/bi_{}.jpg'.format(i[:len(i) - 4]), bi_)
        cv2.imwrite('ocr/resize/bid/bid_{}.jpg'.format(i[:len(i) - 4]), bi_di_)
        cv2.imwrite('ocr/resize/s1/s1_{}.jpg'.format(i[:len(i) - 4]), sharp1)
        cv2.imwrite('ocr/resize/s2/s2_{}.jpg'.format(i[:len(i) - 4]), sharp2)
        cv2.imwrite('ocr/resize/gauss_sh/gsh_{}.jpg'.format(i[:len(i) - 4]),
                    sharp3)
        cv2.imwrite('ocr/resize/er_bi_sh/bsh_{}.jpg'.format(i[:len(i) - 4]),
                    sharp4)
        cv2.imwrite('ocr/resize/median_sh/msh_{}.jpg'.format(i[:len(i) - 4]),
                    sharp5)
        cv2.imwrite('ocr/resize/bid_sh/bidsh_{}.jpg'.format(i[:len(i) - 4]),
                    sharp6)


##########################START#############################
def detect_shape(c):
    shape = ""
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # Triangle
    if len(approx) == 3:
        shape = "triangle"

    # Square or rectangle
    elif len(approx) in (4, ):
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # A square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # Pentagon
    # elif len(approx) == 5:
    #     shape = "pentagon"

    # Otherwise assume as circle or oval
    # elif len(approx) > 8:
    else:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "circle" if ar >= 0.95 and ar <= 1.05 else "oval"

    return shape


##########################END###############################


def text_recognition(imagepath, col):
    word, filename = "", ""
    pred = ''
    data_ = []
    for i in os.listdir(imagepath):
        if i.endswith("png") or i.endswith('jpg'):
            filename = i
            ########################START#######################
            roi_pth = str(imagepath) + "/" + str(i)
            # removing morph_ from beginning of image name
            rm_mrph = '_'.join(i.split('_')[1:])
            # replacing .jpg with .png
            fname = rm_mrph.split('.')[0] + '.png'
            non_resized_roi_pth = 'ocr/ROI/' + fname
            ROI = cv2.imread(non_resized_roi_pth)
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(thresh, kernel, iterations=2)
            cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
            # boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            # cnts, bbx = zip(*sorted(zip(cnts, boundingBoxes),
            #                         key=lambda b: b[1][0], reverse=False))
            # cnts_h = list(zip(cnts, _))
            # for i in range(len(cnts_h)):
            #     for j in range(len(cnts_h)):
            #         x1, y1, w1, h1 = cv2.boundingRect(cnts_h[i][0])
            #         x2, y2, w2, h2 = cv2.boundingRect(cnts_h[j][0])
            #         if x1 < x2:
            #             cnts_h[j], cnts_h[i] = cnts_h[i], cnts[j]

            # cnts, _ = [], []

            # for x in cnts_h:
            #     cnts.append(x[0])
            #     cnts.append(x[1])
            print(colored(non_resized_roi_pth, 'blue'))
            x, y, w, h = cv2.boundingRect(cnts[0])
            if len(cnts) in (2, 3) and _[..., 2][0][0] == 1:
                # if len(cnts) == 2 and _[..., 2][0][0] == 1:
                # cv2.imwrite(
                #     f'ocr/contours/{z}.jpg', cv2.drawContours(ROI.copy(), [cnts[0]], 0, (0, 255, 0), 1))
                # z += 1
                shape1 = detect_shape(cnts[0])
                shape2 = detect_shape(cnts[1])
                # print(colored((shape1, roi_pth), 'green'))
                h1, w1 = ROI.shape[:2]
                x, y, w, h = cv2.boundingRect(cnts[0])
                if shape1 in ('rectangle', 'square'
                              ) and (w * h) / cv2.contourArea(cnts[0]) < 1.6:
                    pred = (shape1, -1)
                elif shape1 in ('circle',
                                'oval') and (w1 * h1) / (w * h) < 1.6:
                    pred = (shape1, -1, x, y, w, h)

                else:
                    pred = main(roi_pth)

            # detecting arrow
            elif len(cnts) == 1 and max(w, h) > 100:
                h1, w1 = ROI.shape[:2]
                shape = detect_shape(cnts[0])
                if shape in ('circle', 'oval') and (w1 * h1) / (w * h) < 1.6:
                    # if shape in ('circle', 'oval') and (w*h)/cv2.contourArea(cnts[0]) < 8:

                    img = cv2.bitwise_not(dilate)

                    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

                    labels, stats = cv2.connectedComponentsWithStats(img,
                                                                     8)[1:3]

                    # for label in np.unique(labels)[1:]:

                    arrow = labels == np.unique(labels)[1:][0]

                    indices = np.transpose(np.nonzero(arrow))  #y,x

                    dist = distance.cdist(indices, indices, 'euclidean')

                    far_points_index = np.unravel_index(
                        np.argmax(dist), dist.shape)  #y,x

                    far_point_1 = indices[far_points_index[0], :]  # y,x
                    far_point_2 = indices[far_points_index[1], :]  # y,x

                    ### Slope
                    arrow_slope = (far_point_2[0] - far_point_1[0]) / (
                        far_point_2[1] - far_point_1[1])
                    arrow_angle = math.degrees(math.atan(arrow_slope))

                    ### Length
                    arrow_length = distance.cdist(far_point_1.reshape(1, 2),
                                                  far_point_2.reshape(1, 2),
                                                  'euclidean')[0][0]

                    ### Thickness
                    x = np.linspace(far_point_1[1], far_point_2[1], 20)
                    y = np.linspace(far_point_1[0], far_point_2[0], 20)
                    line = np.array([[yy, xx] for yy, xx in zip(y, x)])

                    x1, y1 = tuple(line[-1][::-1].astype(int))
                    x2, y2 = tuple(line[0][::-1].astype(int))
                    pred = ('arrow', -1, x1, y1, x2, y2)
                else:
                    pred = main(roi_pth)
            else:
                pred = main(roi_pth)
                # h1, w1 = ROI.shape[:2]
                # x, y, w, h = cv2.boundingRect(cnts[0])
                # r = max(w/2, h/2)
                # area = 3.14*r**2
                # shape = detect_shape(cnts[0])
                # if shape in ('rectangle', 'square') and (w1*h1)/(w*h) < 1.6:
                #     pred = (shape, -1)
                # elif shape in ('circle', 'oval') and area/cv2.contourArea(cnts[0]) < 1.6:
                #     # elif shape in ('circle', 'oval') and (w1*h1)/(w*h) < 1.6 and area/cv2.contourArea(cnts[0]) < 1.6:
                #     pred = (shape, -1, x, y, w, h)
                # else:
                #     pred = main(roi_pth)
            ########################END############################
            '''
            pred = main(str(imagepath) +"/"+ str(i))
            '''
            data_.append((filename, pred))
    return data_


def text_init(image, impth, col):
    H, W = cv2.imread('ocr/outputs/basic_input.jpg').shape[:2]
    df = extractROI(image)
    scale_img()
    dilate_erode(col)

    pre = " "
    if impth.endswith("di_"):
        pre = "di_"
    elif impth.endswith("de"):
        pre = "Erode_"
    elif impth.endswith("de1"):
        pre = "Erode_"
    elif impth.endswith("median"):
        pre = "median_"
    elif impth.endswith("gauss"):
        pre = "gauss_"
    elif impth.endswith("bi_"):
        pre = "bi_"
    elif impth.endswith("bifilter"):
        pre = "bi_"
    elif impth.endswith("s1"):
        pre = "s1_"
    elif impth.endswith("s2"):
        pre = "s2_"
    elif impth.endswith("gauss_sh"):
        pre = "gsh_"
    elif impth.endswith("median_sh"):
        pre = "msh_"
    elif impth.endswith("er_bi_sh"):
        pre = "bsh_"
    elif impth.endswith("bid_sh"):
        pre = "bidsh_"
    elif impth.endswith("bid"):
        pre = "bid_"
    elif impth.endswith("morph"):
        pre = "morph_"
    elif impth.endswith("red"):
        pre = "red_"

    txt_col = ()
    if col == 'blue':
        txt_col = (255, 0, 0)
    elif col == 'green':
        txt_col = (36, 255, 12)
    elif col == 'red':
        txt_col = (0, 0, 255)
    elif col == 'black':
        txt_col = (0, 0, 0)

    extimg = cv2.imread("ocr/outputs/basic_input.jpg")
    gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bbox_ = None
    x, y = None, None
    text_data = text_recognition(impth, col)
    # print(text_data)
    ROI_number, count = 0, 0
    # read = []
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > 10000:
    rec_text = ''
    rec_data = []
    for d in text_data:
        # print("entered in text data loop")
        '''
        for e, p, q, r, s in zip(df['ImageName'], df['x'], df['y'], df['x+w'], df['y+h']):
        '''
        ############################START###############################
        for e, p, q, r, s, dot_x, dot_y in zip(df['ImageName'], df['x'],
                                               df['y'], df['x+w'], df['y+h'],
                                               df['dot_coordinate_x'],
                                               df['dot_coordinate_y']):
            ############################END#################################
            # print("entered to iterate csv")
            # print(pre+e)
            if d[0] == pre + e:
                word = d[1][0]
                ######################START#########################
                confidence = d[1][1]
                if confidence == -1:
                    # draw shape
                    if word in ('rectangle', 'square'):
                        text = cv2.rectangle(extimg, (p + 10, q + 10),
                                             (r - 10, s - 10), txt_col,
                                             round(8 / 4032 * W))
                    elif word in ('circle', 'oval'):
                        # print(colored((p, q, r, s), 'blue'))
                        # w, h = r-p, s-q
                        cnt_x, cnt_y = d[1][2], d[1][3]
                        cnt_w, cnt_h = d[1][4], d[1][5]
                        r = max(cnt_w / 2, cnt_h / 2)
                        text = cv2.circle(extimg, (int(p + cnt_x + cnt_w / 2),
                                                   int(q + cnt_y + cnt_h / 2)),
                                          int(r), txt_col, round(8 / 4032 * W))
                    elif word in ('arrow', ):
                        text = cv2.arrowedLine(extimg,
                                               (p + d[1][2], q + d[1][3]),
                                               (p + d[1][4], q + d[1][5]),
                                               txt_col,
                                               round(8 / 4032 * W),
                                               tipLength=0.2)
                    rec_data.append(word)
                else:
                    print("WORD :: ", word)
                    char_set = '!@#$%^&*()_+=-,./<>?{}|\][ '
                    if col == "blue":
                        for i in char_set:
                            if word[0] == i:
                                # print("REC:", rec_text)
                                checked_text = ''
                                if len(word) != 1:
                                    checked_text = spellchecker.SpellChecker(
                                    ).correction(word[1:])
                                rec_text = word[0] + checked_text
                                break

                            else:
                                rec_text = 'none'
                                # print("REC:", rec_text)
                                checked_text = word
                                # len of word and contour is more than 1
                                ROI = cv2.imread(f'ocr/resize/morph/{d[0]}')
                                gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                                _, thresh = cv2.threshold(
                                    gray, 128, 255, cv2.THRESH_BINARY_INV)
                                cnts = cv2.findContours(
                                    thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
                                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                                if len(word) != 1:
                                    if len(cnts) != 1:
                                        checked_text = spellchecker.SpellChecker(
                                        ).correction(word)
                                    else:
                                        checked_text = word[0]
                                # checked_text = word
                                rec_text = checked_text
                                # print("ELSE :",rec_text)
                    if col == "red":
                        rec_text = word
                    rec_data.append(rec_text)
                    print("RECOGNISED DATA LIST : ", rec_data)
                    # print("CHECKED_TEXT :",rec_text)

                    ##################### Fixing the text position ##################
                    fname = '_'.join(
                        d[0].split('_')[1:]).split('.')[0] + '.png'
                    img = cv2.imread(f'ocr/ROI/{fname}')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 128, 255,
                                              cv2.THRESH_BINARY_INV)
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                    left_cnt = sorted(cnts,
                                      key=lambda x: cv2.boundingRect(x)[0])[0]
                    bottom_cnt = sorted(cnts,
                                        key=lambda x: cv2.boundingRect(x)[1] +
                                        cv2.boundingRect(x)[3])[-1]

                    X = cv2.boundingRect(left_cnt)[0]
                    Y = cv2.boundingRect(bottom_cnt)[1] + cv2.boundingRect(
                        bottom_cnt)[3]
                    ############################# END ##################################

                    # finding the width and height of the image to resize text
                    # based on it's dimension
                    H, W = cv2.imread('ocr/outputs/basic_input.jpg').shape[:2]

                    if rec_text[0] == '.':
                        rec_text = rec_text[0] + rec_text[1:].strip()
                        text = cv2.putText(extimg, rec_text,
                                           (int(dot_x - (5 * 3 / 4032 * W)),
                                            int(dot_y + 3 / 4032 * W)),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           3 / 4032 * W, txt_col,
                                           round(8 / 4032 * W))
                    else:
                        text = cv2.putText(extimg, rec_text, (p + X, q + Y),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           3 / 4032 * W, txt_col,
                                           round(8 / 4032 * W))
                    # fname = e.split('.')[0]+'.png'
                    # h, w = cv2.imread(f'ocr/ROI/{fname}').shape[:2]
                    # fontScale = w/(18*len(rec_text))

                    # if rec_text[0] == '.':
                    #     # rec_text = rec_text[0]+rec_text[1:].strip()
                    #     text = cv2.putText(extimg, rec_text, (int(dot_x-5*3), int(dot_y+3)),
                    #                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, txt_col, round(2.5*fontScale))
                    # else:
                    #     text = cv2.putText(extimg, rec_text, (p+10, int(q+(h/1.3))),
                    #                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, txt_col, round(2.5*fontScale))
                ############################END###############################
                '''
                print("WORD :: ", word)
                char_set = '!@#$%^&*()_+=-,./<>?{}|\][ '
                if col == "blue":
                    for i in char_set:
                        if word[0] == i:
                            # print("REC:", rec_text)
                            checked_text = spellchecker.SpellChecker().correction(word[1:])
                            rec_text = word[0] + checked_text
                            break

                        else:
                            rec_text = 'none'
                            # print("REC:", rec_text)
                            checked_text = spellchecker.SpellChecker().correction(word)
                            rec_text = checked_text
                            # print("ELSE :",rec_text)
                if col == "red":
                    rec_text = word
                rec_data.append(rec_text)
                print("RECOGNISED DATA LIST : ",rec_data)
                # print("CHECKED_TEXT :",rec_text)
                text = cv2.putText(extimg,rec_text,(p+10 ,q+70),cv2.FONT_HERSHEY_SIMPLEX, 3,txt_col,8)
                '''
                # cv2.imwrite('ocr/outputs/put_text_{}.jpg'.format(col),text)
                cv2.imwrite('ocr/outputs/basic_input.jpg'.format(col), text)
                count += 1

    print("text_data info : ", text_data)
    return (df, rec_data, text_data)


def run():
    in_time = time.time()
    img_files = []
    ROI_coordinates = None
    for i in os.listdir("ocr/outputs"):
        if i.startswith("i") or i.startswith("bl_"):
            img_files.append(i)
    print(img_files)
    for j in img_files:
        print(j)
        if j.endswith("ib.jpg"):
            color = "blue"
            print(j)
            '''
            ROI_coordinates = text_init("ocr/outputs/"+j,"ocr/resize/morph",color)
            '''
            ROI_coordinates = blue.text_init("ocr/outputs/" + j,
                                             "ocr/resize/morph", color)

        if j.endswith("ir.jpg"):
            color = "red"
            ROI_coordinates = text_init("ocr/outputs/"+j,"ocr/resize/red",color)
        else:
            print("X --- X --- X")
    return ROI_coordinates
