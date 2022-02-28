

import cv2, os,csv , time
import numpy as np
# import keras_ocr
import matplotlib.pyplot as plt
from spellchecker import spellchecker
import pandas as pd
from PIL import Image, ImageEnhance
import main

# pred = main.main("/home/rohit/Downloads/Skype/tress4.png")

# pipeline = keras_ocr.pipeline.Pipeline()
# def try_rec():
#     print("Entered in TRY_RECOGINITION METHOD")
#     main.main("/home/rohit/Downloads/Skype/tress4.png")
# try_rec()

def rgb(img):
    image = cv2.imread(img, -1)
    # image = cv2.rotate(image,cv2.cv2.ROTATE_90_CLOCKWISE)
    bhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bmask = cv2.inRange(bhsv, np.array([110, 50, 50]), np.array([130, 255, 255]))
    # print(j)
    ### finding red color only
    rhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rmask = cv2.inRange(rhsv, np.array([165, 155, 84]), np.array([179, 255, 255]))
    # rmask = cv2.inRange(rhsv,np.array([0,20 ,60]),np.array([60,180,255]))
    # gmask = cv2.inRange(rhsv, np.array([40, 20, 50]), np.array([100, 100, 180]))    ## for pdf images
    gmask = cv2.inRange(rhsv, np.array([40, 60, 25]), np.array([100, 255, 255]))  ## for normal image
    # gmask = cv2.inRange(rhsv,np.array([65,60,60]), np.array([80, 255,255]))
    inv_bmask = cv2.bitwise_and(image, image, mask=bmask)
    kernel = np.ones((1, 1), np.uint8)
    _, ir = cv2.threshold(rmask, 50, 200, cv2.THRESH_BINARY_INV)
    ir = cv2.erode(ir,kernel,iterations=1)
    cv2.imwrite("../outputs/ir.jpg", ir)
    _, ib = cv2.threshold(bmask, 50, 200, cv2.THRESH_BINARY_INV)
    # ib = cv2.erode(ib, kernel, iterations=1)
    cv2.imwrite("../outputs/ib.jpg", ib)
    _, ig = cv2.threshold(gmask, 20, 250, cv2.THRESH_BINARY_INV)
    # ig = cv2.erode(ig, kernel, iterations=1)
    cv2.imwrite("../outputs/ig.jpg", ig)

    ### For black
    # black_mask = cv2.inRange(bhsv, np.array([0, 0, 0]), np.array([150, 30, 150]))
    black_mask = cv2.inRange(bhsv, np.array([0, 0, 0]), np.array([100, 185, 60]))
    res_bl = cv2.bitwise_and(image, image, mask=black_mask)
    _, iblack = cv2.threshold(res_bl, 5, 250, cv2.THRESH_BINARY_INV)  ## inverted green
    cv2.imwrite("../outputs/bl_text.jpg", iblack)

# def extractROI(image):
#     # Load image, grayscale, Gaussian blur, adaptive threshold
#     extimg = cv2.imread(image)
#     gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9,9), 0)
#     thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
#
#     # Dilate to combine adjacent text contours
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
#     dilate = cv2.dilate(thresh, kernel, iterations=4)
#
#     # Find contours, highlight text areas, and extract ROIs
#     cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#     ROI_number = 0
#
#     textColor = " "
#
#     if image.endswith("ib.jpg"):
#         textColor = "Blue"
#     if image.endswith("ir.jpg"):
#         textColor = "Red"
#     if image.endswith("ig.jpg"):
#         textColor = "Green"
#     if image.endswith("bl.jpg"):
#         textColor = "Black"
#
#     df = pd.DataFrame(
#         {"ImageName": [],
#          "y-20": [],
#          "y+h": [],
#          "x-60": [],
#          "x+w+60": [],
#          "TextColor": []
#          },dtype=int
#     )
#     with open("../r.csv", 'w') as cfile:
#         for c in cnts:
#             area = cv2.contourArea(c)
#             if area > 10000:
#                 x,y,w,h = cv2.boundingRect(c)
#                 left = y-50, y+h
#                 right = x-60, x+w+70
#
#                 # pair = [left,right]
#                 ## creating a dataframe
#                 df2 = pd.DataFrame(
#                     {   "ImageName":["ROI_{}.jpg".format(ROI_number)],
#                         "y-20":[y-20],
#                         "y+h":[y+h],
#                         "x-60":[x-60],
#                         "x+w+60":[x+w+60],
#                         "TextColor":[textColor]
#                     }
#                 )
#
#                 df = df.append(df2)
#
#                 if textColor == "Blue":
#                     ROI = extimg[left[0]:left[1], right[0]-60:right[1]-50]
#                     cv2.imwrite('../ROI/ROI_{}.png'.format(ROI_number), ROI)
#                     ROI_number += 1
#                 else:
#                     ROI = extimg[left[0]:left[1], right[0]+30:right[1]-30]
#                     cv2.imwrite('../ROI/ROI_{}.png'.format(ROI_number), ROI)
#                     ROI_number += 1
#     print(df)
#     return df


ROI_number = 0

def extractROI(image):
    extimg = cv2.imread(image)
    gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=6)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    ROI_number = 0
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
         "y-20": [],
         "y+h": [],
         "x-60": [],
         "x+w+60": [],
         "TextColor": []
         },dtype=int
    )
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 5000:
            x,y,w,h = cv2.boundingRect(c)
            left = y, y+h
            right = x, x+w

            # pair = [left,right]
            ## creating a dataframe
            df2 = pd.DataFrame(
                {   "ImageName":["ROI_{}.jpg".format(ROI_number)],
                    "y-20":[y],
                    "y+h":[y+h],
                    "x-60":[x],
                    "x+w+60":[x+w],
                    "TextColor":[textColor]
                }
            )

            df = df.append(df2)
            ROI = extimg[left[0]:left[1], right[0]:right[1]]
            cv2.imwrite('../ROI/ROI_{}.png'.format(ROI_number), ROI)
            ROI_number += 1
    # print(df)
    return df

def scale_img():
    files = []
    for i in os.listdir("../ROI"):
        if i.endswith('png'):
            files.append(i)
    files.sort()
    # print(files)
    c = 0
    for j in files:
        im = cv2.imread("../ROI/" + str(j), -1)
        # resize_image = cv2.resize(im, (350, 130))  ### 275 -- 110
        resize_image = cv2.resize(im, (300, 100))  ### 275 -- 110
        cv2.imwrite("../resize/ROI_{}.png".format(c), resize_image)
        # print(j,os.path.exists("ROI/" + str(j)))
        c+=1

# def dilate_erode():
#     ## DILATION & EROSION
#     files = []
#     for j in os.listdir("../resize"):
#         if j.endswith("png"):
#             files.append(j)
#     files.sort()
#     # print("fnames : ", files)
#
#     for i in files:
#         img_ = cv2.imread('../resize/' + str(i), -1)
#         #         print(img_)
#         # Taking a matrix of size 5 as the kernel
#         kernel = np.ones((3, 3), np.uint8)
#         kernel_ = np.ones((5, 5), np.uint8)
#         #         plt.figure(figsize=(9,9))
#         #         plt.imshow(img_)
#         img_erosion = cv2.erode(img_, kernel, iterations=2)
#         img_er = cv2.erode(img_, kernel, iterations=1)
#         img_dilation = cv2.dilate(img_, kernel_, iterations=1)
#         er2 = cv2.erode(img_dilation, kernel, iterations=1)
#
#         kernelsh = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#         sharp1 = cv2.filter2D(img_erosion, -1, kernelsh)
#
#         di2 = cv2.dilate(img_erosion, kernel_, iterations=1)
#         sharp2 = cv2.filter2D(di2,-1,kernelsh)
#
#         gausBlur = cv2.GaussianBlur(img_, (5, 5), 0)
#         medBlur = cv2.medianBlur(img_, 3)
#         bilFilter = cv2.bilateralFilter(img_, 9, 75, 75)
#
#         bi_ = cv2.bilateralFilter(img_erosion,9, 75, 75)
#         bi_di_ = cv2.bilateralFilter(di2,9, 75, 75)
#
#         sharp3 = cv2.filter2D(gausBlur,-1,kernelsh)
#         sharp4 = cv2.filter2D(bi_,-1,kernelsh)
#         sharp5 = cv2.filter2D(medBlur,-1,kernelsh)
#         sharp6 = cv2.filter2D(bi_di_,-1,kernelsh)
#
#
#         gray_mo = cv2.cvtColor(sharp2, cv2.COLOR_BGR2GRAY)
#         thresh_mo = cv2.threshold(gray_mo, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#         # Morph open to remove noise
#         kernel_mo = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#         opening_mo = cv2.morphologyEx(thresh_mo, cv2.MORPH_OPEN, kernel_mo, iterations=1)
#         # Find contours and remove small noise
#         cnts_mo = cv2.findContours(opening_mo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts_mo = cnts_mo[0] if len(cnts_mo) == 2 else cnts_mo[1]
#         for c in cnts_mo:
#             area_mo = cv2.contourArea(c)
#             if area_mo < 10:
#                 cv2.drawContours(opening_mo, [c], -1, 0, -1)
#
#         # Invert and apply slight Gaussian blur
#         result_mo = 255 - opening_mo
#
#         img_dilation_mo = cv2.erode(result_mo, kernel_mo, iterations=1)
#
#         img_dilation_mo = cv2.dilate(img_dilation_mo, kernel_mo, iterations=1)
#
#         result_mo = img_dilation_mo
#         # result = cv2.GaussianBlur(result, (7,7), 0)
#         result_mo = cv2.GaussianBlur(result_mo, (3, 3), 0)
#
#         # cv2.imwrite('Input', img)
#         cv2.imwrite('../resize/erode/Erode_{}.jpg'.format(i[:len(i) - 4]), img_erosion)
#         cv2.imwrite('../resize/erode1/Erode_{}.jpg'.format(i[:len(i) - 4]), img_er)
#         cv2.imwrite('../resize/dilate/Dilate_{}.jpg'.format(i[:len(i)-4]), img_dilation)
#         cv2.imwrite('../resize/gauss/gauss_{}.jpg'.format(i[:len(i)-4]), gausBlur)
#         cv2.imwrite('../resize/median/median_{}.jpg'.format(i[:len(i)-4]), medBlur)
#         cv2.imwrite('../resize/bifilter/bi_{}.jpg'.format(i[:len(i)-4]), bilFilter)
#         cv2.imwrite('../resize/di_/di_{}.jpg'.format(i[:len(i) - 4]), di2)
#         cv2.imwrite('../resize/er_/er_{}.jpg'.format(i[:len(i)-4]), er2)
#         cv2.imwrite('../resize/er_bi_/bi_{}.jpg'.format(i[:len(i)-4]), bi_)
#         cv2.imwrite('../resize/bid/bid_{}.jpg'.format(i[:len(i)-4]), bi_di_)
#         cv2.imwrite('../resize/s1/s1_{}.jpg'.format(i[:len(i)-4]), sharp1)
#         cv2.imwrite('../resize/s2/s2_{}.jpg'.format(i[:len(i)-4]), sharp2)
#         cv2.imwrite('../resize/gauss_sh/gsh_{}.jpg'.format(i[:len(i)-4]), sharp3)
#         cv2.imwrite('../resize/er_bi_sh/bsh_{}.jpg'.format(i[:len(i)-4]), sharp4)
#         cv2.imwrite('../resize/median_sh/msh_{}.jpg'.format(i[:len(i)-4]), sharp5)
#         cv2.imwrite('../resize/bid_sh/bidsh_{}.jpg'.format(i[:len(i)-4]), sharp6)
#         cv2.imwrite('../resize/morph/morph_{}.jpg'.format(i[:len(i)-4]), sharp6)

def dilate_erode():
    ## DILATION & EROSION
    files = []
    for j in os.listdir("../resize"):
        if j.endswith("png"):
            files.append(j)
    files.sort()
    # print("fnames : ", files)

    for i in files:
        img_ = cv2.imread('../resize/' + str(i), -1)
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
        sharp2 = cv2.filter2D(di2,-1,kernelsh)

        gausBlur = cv2.GaussianBlur(img_, (5, 5), 0)
        medBlur = cv2.medianBlur(img_, 3)
        bilFilter = cv2.bilateralFilter(img_, 9, 75, 75)

        bi_ = cv2.bilateralFilter(img_erosion,9, 75, 75)
        bi_di_ = cv2.bilateralFilter(di2,9, 75, 75)

        sharp3 = cv2.filter2D(gausBlur,-1,kernelsh)
        sharp4 = cv2.filter2D(bi_,-1,kernelsh)
        sharp5 = cv2.filter2D(medBlur,-1,kernelsh)
        sharp6 = cv2.filter2D(bi_di_,-1,kernelsh)

        image = cv2.imread('../resize/' + str(i), -1)

        image = cv2.resize(image, (600, 150))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours and remove small noise
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # cv2.imwrite('Input', img)
        cv2.imwrite('../resize/erode/Erode_{}.jpg'.format(i[:len(i) - 4]), img_erosion)
        cv2.imwrite('../resize/erode1/Erode_{}.jpg'.format(i[:len(i) - 4]), img_er)
        cv2.imwrite('../resize/dilate/Dilate_{}.jpg'.format(i[:len(i)-4]), img_dilation)
        cv2.imwrite('../resize/gauss/gauss_{}.jpg'.format(i[:len(i)-4]), gausBlur)
        cv2.imwrite('../resize/median/median_{}.jpg'.format(i[:len(i)-4]), medBlur)
        cv2.imwrite('../resize/bifilter/bi_{}.jpg'.format(i[:len(i)-4]), bilFilter)
        cv2.imwrite('../resize/di_/di_{}.jpg'.format(i[:len(i) - 4]), di2)
        cv2.imwrite('../resize/er_/er_{}.jpg'.format(i[:len(i)-4]), er2)
        cv2.imwrite('../resize/er_bi_/bi_{}.jpg'.format(i[:len(i)-4]), bi_)
        cv2.imwrite('../resize/bid/bid_{}.jpg'.format(i[:len(i)-4]), bi_di_)
        cv2.imwrite('../resize/s1/s1_{}.jpg'.format(i[:len(i)-4]), sharp1)
        cv2.imwrite('../resize/s2/s2_{}.jpg'.format(i[:len(i)-4]), sharp2)
        cv2.imwrite('../resize/gauss_sh/gsh_{}.jpg'.format(i[:len(i)-4]), sharp3)
        cv2.imwrite('../resize/er_bi_sh/bsh_{}.jpg'.format(i[:len(i)-4]), sharp4)
        cv2.imwrite('../resize/median_sh/msh_{}.jpg'.format(i[:len(i)-4]), sharp5)
        cv2.imwrite('../resize/bid_sh/bidsh_{}.jpg'.format(i[:len(i)-4]), sharp6)
        cv2.imwrite('../resize/morph/morph_{}.jpg'.format(i[:len(i)-4]), result)


def text_recognition(imagepath,col):
    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.

    word ,filename = "",""
    pred  = ''
    data_ = []

    # for i in os.listdir(imagepath):
    #     print(i,type(i))

    for i in os.listdir(imagepath):
        if i.endswith("png") or i.endswith('jpg'):
            # # Get a set of three example images
            # images = [keras_ocr.tools.read(url) for url in [str(imagepath) +"/"+ str(i)]]
            print(f"File Name : {str(imagepath)+'/'+str(i)} ------   TextColor : {col}")

            if os.path.isdir(str(imagepath)):
                print("yes found")
            else:
                print("NOT FOUND")
            filename = i
            pred = main.main(str(imagepath) +"/"+ str(i))
            # pred = main.main("/home/rohit/Downloads/Skype/tress4.png")
            # main.main("/home/rohit/Downloads/Skype/tress4.png")
            # # Each list of predictions in prediction_groups is a list of
            # # (word, box) tuples.
            # prediction_groups = pipeline.recognize(images)
            # # Plot the predictions
            # fig, axs = plt.subplots(nrows=len(images) + 1, figsize=(20, 20))
            # for ax, image, predictions in zip(axs, images, prediction_groups):
            #     #     pred = spell.correction(predictions[:1])
            #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
            #     pred = predictions
            #
            # for w in pred:
            #     word = str(w[0])
            # word = spellchecker.SpellChecker().correction(word)
            data_.append((filename,pred))
    return data_

def text_init(image,impth,col):
    df = extractROI(image)
    scale_img()
    dilate_erode()

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
    elif impth.endswith("bi_") :
        pre = "bi_"
    elif impth.endswith("bifilter") :
        pre = "bi_"
    elif impth.endswith("s1") :
        pre = "s1_"
    elif impth.endswith("s2") :
        pre = "s2_"
    elif impth.endswith("gauss_sh") :
        pre = "gsh_"
    elif impth.endswith("median_sh") :
        pre = "msh_"
    elif impth.endswith("er_bi_sh") :
        pre = "bsh_"
    elif impth.endswith("bid_sh") :
        pre = "bidsh_"
    elif impth.endswith("bid") :
        pre = "bid_"
    elif impth.endswith("morph") :
        pre = "morph_"

    txt_col = ()
    if col == 'blue':
        txt_col = (255, 0, 0)
    elif col == 'green':
        txt_col = (36,255,12)
    elif col == 'red':
        txt_col = (0, 0, 255)
    elif col == 'black':
        txt_col = (0, 0, 0)

    extimg = cv2.imread(image)
    gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bbox_ = None
    x, y = None, None
    text_data = text_recognition(impth,col)
    # print(text_data)
    ROI_number ,count = 0,0
    # read = []
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > 10000:
    for d in text_data:
        # print("entered in text data loop")
        for e, p, q, r, s in zip(df['ImageName'], df['x-60'], df['y-20'], df['x+w+60'], df['y+h']):
            # print("entered to iterate csv")
            # print(pre+e)
            rec_text = ''
            if d[0] == pre+e:
                word = d[1][0]
                char_set = '!@#$%^&*()_+=-,./<>?{}|\][ '
                for i in char_set:
                    if word[0] == i:
                        print("REC:", rec_text)
                        checked_text = spellchecker.SpellChecker().correction(word[1:])
                        rec_text = word[0] + checked_text
                        break
                    else:
                        rec_text = 'none'
                        print("REC:", rec_text)
                        checked_text = spellchecker.SpellChecker().correction(word)
                        rec_text = checked_text
                        print("ELSE :",rec_text)

                # print("CHECKED_TEXT :",rec_text)
                text = cv2.putText(extimg,rec_text,(p+50 ,q+110),cv2.FONT_HERSHEY_SIMPLEX, 3,txt_col,7)
                cv2.imwrite('../outputs/put_text_{}.jpg'.format(col),text)
                count += 1

def run():
    in_time = time.time()
    img_files = []

    for i in os.listdir("../outputs"):
        if i.startswith("i") or i.startswith("bl_"):
            img_files.append(i)
    print(img_files)
    for j in img_files:
        if j.endswith("ib.jpg"):
            color = "blue"
            text_init("../outputs/"+j,"../resize/morph",color)
        # if j.endswith("ir.jpg"):
        #     print("{} image is in process".format(j))
        #     color = "red"
        #     text_init("../outputs/"+j,"../resize/morph",color)
        # if j.endswith("ig.jpg"):
        #     print("{} image is in process".format(j))
        #     color = "green"
        #     text_init("../outputs/"+j,"../resize/bid_sh",color)
        # if j.endswith("bl_text.jpg"):
        #     color = "black"
        #     text_init("../outputs/"+j,"../resize/gauss_sh",color)
        else:
            print("Sorry.. Invalid File")
    print(time.time() - in_time)
