import cv2
import os
import math
import numpy as np
from spellchecker import spellchecker
import pandas as pd
from .main import main
from scipy.spatial import distance
from termcolor import colored


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

    else:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "circle" if ar >= 0.95 and ar <= 1.05 else "oval"

    return shape


def extractROI(image):

    # reading image having extracted text for blue color
    # and preprocessing on it to extract ROIs
    # original_img -> grayed -> blurred -> binarized -> dilated(itrs=8)
    extimg = cv2.imread(image)
    gray = cv2.cvtColor(extimg, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=8)

    # finding contours of dilated image
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initializing dataframe
    df = pd.DataFrame(
        {
            "ImageName": [],
            "y": [],
            "y+h": [],
            "x": [],
            "x+w": [],
            "TextColor": [],
            "dot_coordinate_x": [],
            "dot_coordinate_y": [],
        },
        dtype=int,
    )

    # normalizing factor to retain settings for images having varying dimensions
    W = extimg.shape[1]
    F = W / 4032

    ROI_number = 0

    for c in cnts:
        area = cv2.contourArea(c)
        const = 3500 * F
        if area <= const:  # ignoring contours having smaller area
            continue

        # finding coordinates of contour boundary and extracting ROI
        x, y, w, h = cv2.boundingRect(c)
        top, bottom = y, y + h
        left, right = x, x + w

        ROI = extimg[top:bottom, left:right]
        kernel = np.ones((3, 3), np.uint8)
        ROI = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel)

        ROIs = []  # list to store shape and text ROIs after seperating from main ROI

        # finding contours of main ROI
        grayed = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        threshed = cv2.threshold(grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
        cntrs = cv2.findContours(
            threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(cntrs) > 1:  # seperating shape from main ROI as it has more than 1 contour

            # calculating distance between shape and text contour and seperating them
            cntrs = sorted(cntrs, key=lambda x: cv2.boundingRect(x)[0])
            cntr1 = cntrs[0]
            rightmost_point_cntr1 = max(cntr1, key=lambda x: x[0][0])
            cntr2 = cntrs[1]
            leftmost_point_cntr2 = min(cntr2, key=lambda x: x[0][0])
            diff = leftmost_point_cntr2[0][0]-rightmost_point_cntr1[0][0]
            right_padding_cntr1 = diff//2

            w1 = rightmost_point_cntr1[0][0]+right_padding_cntr1
            x1, y1, w1, h1 = x, y, w1, h  # bounding rect of left roi

            x2, y2, w2, h2 = x+w1, y, w-w1, h  # bounding rect of right roi

            roi_left = ROI[:, :w1]
            roi_right = ROI[:, w1:]

            # filling gaps between strokes in distorted shape
            grayed = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
            threshed = cv2.threshold(
                grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
            cntrs = cv2.findContours(
                threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            eroded = roi_left.copy()
            kernel = np.ones((2, 2), np.uint8)
            len_cntrs = len(cntrs)
            iters_left = 3
            while len_cntrs != 2 and iters_left:
                eroded = cv2.erode(eroded, kernel)
                grayed = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
                threshed = cv2.threshold(
                    grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
                cntrs = cv2.findContours(
                    threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                len_cntrs = len(cntrs)
                iters_left -= 1
            roi_left = eroded

            grayed = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
            threshed = cv2.threshold(
                grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
            cntrs = cv2.findContours(
                threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cntrs = sorted(cntrs, key=lambda x: cv2.boundingRect(x)[0])
            cntr = cntrs[0]
            x3, y3, w3, h3 = cv2.boundingRect(cntr)
            if detect_shape(cntr) and h3 >= 98 and (w3 * h3) / cv2.contourArea(cntr) < 1.6:
                ROIs.append((roi_left, x1, y1, w1, h1))
                ROIs.append((roi_right, x2, y2, w2, h2))
            else:
                ROIs.append((ROI, x, y, w, h))

        else:  # ROI having single contour doesn't contain both shape and text
            ROIs.append((ROI, x, y, w, h))

        for ROI, x, y, w, h in ROIs:
            ################## find dots' coordinates ##################

            # Load image, grayscale, Otsu's threshold
            image = ROI.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            final_x, final_y = 0, 0  # initializing coordinate point of dot
            for c in cnts:
                x1, y1, w1, h1 = cv2.boundingRect(c)
                area = w1 * h1  # area of ROI
                area1 = cv2.contourArea(c)  # area of contour

                # filtering dots
                if x1 < 50 and area > (48 * F) and area < (
                        650 * F) and area1 != 0 and area / area1 <= 2:
                    center_x = x1 + w1 / 2
                    center_y = y1 + h1 / 2
                    final_x = x + center_x
                    final_y = y + center_y

            textColor = 'Blue'

            # updating dataframe with values found above
            df2 = pd.DataFrame({
                "ImageName": ["ROI_{}.jpg".format(ROI_number)],
                "y": [y],
                "y+h": [y + h],
                "x": [x],
                "x+w": [x + w],
                "TextColor": [textColor],
                "dot_coordinate_x": [final_x],
                "dot_coordinate_y": [final_y],
            })

            df.loc[len(df.index)] = [
                "ROI_{}.jpg".format(ROI_number), y, y + h, x, x + w, textColor,
                final_x, final_y
            ]

            cv2.imwrite("ocr/ROI/ROI_{}.jpg".format(ROI_number), ROI)

            ROI_number += 1

    return df


def scale_img():
    files = []
    for i in os.listdir("ocr/ROI"):
        if i.endswith("jpg"):
            files.append(i)

    def number(f):
        num = int(f.split("_")[1].split(".")[0])
        return num

    files.sort(key=number)

    c = 0
    for j in files:
        im = cv2.imread("ocr/ROI/" + str(j))
        h, w = im.shape[:2]
        W = 300
        aspect_ratio = W/w
        dim = (W, round(h*aspect_ratio))
        resized = cv2.resize(im, dim)
        resized = 255-resized
        h, w = resized.shape[:2]
        # M = cv2.getRotationMatrix2D((0, 0), -5, 1.0)
        M = cv2.getRotationMatrix2D((0, 0), 0, 1.0)
        rotated = 255-cv2.warpAffine(resized, M, (w, h))
        dilated = rotated.copy()
        grayed = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        threshed = cv2.threshold(grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
        print(colored(threshed.shape, 'red'))
        cntrs = cv2.findContours(
            threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        prev_cntrs_len = len(cntrs)
        kernel = np.ones((2, 2), np.uint8)
        iters_left = 3
        while True:
            prev_dilated = dilated.copy()
            dilated = cv2.dilate(dilated, kernel)
            grayed = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
            threshed = cv2.threshold(
                grayed, 128, 255, cv2.THRESH_BINARY_INV)[1]
            cntrs = cv2.findContours(
                threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(cntrs) != prev_cntrs_len or not iters_left:
                break
            iters_left -= 1
        # h, w = im.shape[:2]
        # if w < 400:
        #     resize_image = cv2.resize(im, (250, 100))
        # else:
        #     resize_image = cv2.resize(im, (500, 100))

        # cv2.imwrite("ocr/resize/morph/morph_ROI_{}.jpg".format(c),
        #             resize_image)
        cv2.imwrite("ocr/resize/morph/morph_ROI_{}.jpg".format(c),
                    prev_dilated)
        c += 1


def text_recognition(imagepath, col):
    word, filename = "", ""
    pred = ""
    data_ = []
    for i in os.listdir(imagepath):
        if i.endswith("png") or i.endswith("jpg"):
            filename = i
            roi_pth = str(imagepath) + "/" + str(i)
            # removing morph_ from beginning of image name
            rm_mrph = "_".join(i.split("_")[1:])
            # replacing .jpg with .png
            fname = rm_mrph.split(".")[0] + ".jpg"
            non_resized_roi_pth = "ocr/ROI/" + fname
            print(colored(non_resized_roi_pth, "green"))
            ROI = cv2.imread(non_resized_roi_pth)
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(thresh, kernel, iterations=1)
            cnts, _ = cv2.findContours(
                dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

            shape = detect_shape(cnts[0])
            x, y, w, h = cv2.boundingRect(cnts[0])
            # h1, w1 = ROI.shape[:2]
            # if shape in ("rectangle", "square") and (w * h) / cv2.contourArea(cnts[0]) < 1.6:
            if shape in ("rectangle", "square") and h >= 98 and (w * h) / cv2.contourArea(cnts[0]) < 1.6:
                pred = (shape, -1, x, y, w, h)

            # elif shape in ("circle", "oval") and (w1 * h1) / (w * h) < 1.63:
            elif shape in ("circle", "oval") and h >= 98 and (w * h) / cv2.contourArea(cnts[0]) < 1.6:
                pred = (shape, -1, x, y, w, h)

            else:
                kernel = np.ones((3, 3), np.uint8)
                open = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel)
                blur = cv2.blur(open, (3, 3))
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 128, 255,
                                       cv2.THRESH_BINARY_INV)[1]
                cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(cntrs) == 1:
                    old_y = cntrs[0][0][0][1]
                    old_x = cntrs[0][0][0][0]
                    down_pts = 0
                    up_pts = 0
                    left_pts = 0
                    right_pts = 0
                    left_arr = []
                    right_arr = []
                    down_arr = []
                    up_arr = []
                    flag = ""
                    flag_x = ""
                    for i, c in enumerate(cntrs[0][1:]):
                        x, y = c[0]
                        if y > old_y:
                            if flag == "in_less":
                                up_arr.append(up_pts)
                                up_pts = 0
                            flag = "in_more"
                            down_pts += 1
                            old_y = y
                        elif y < old_y:
                            if flag == "in_more":
                                down_arr.append(down_pts)
                                down_pts = 0
                            flag = "in_less"
                            up_pts += 1
                            old_y = y

                        if x > old_x:
                            if flag_x == "in_less_x":
                                left_arr.append(left_pts)
                                left_pts = 0
                            flag_x = "in_more_x"
                            right_pts += 1
                            old_x = x
                        elif x < old_x:
                            if flag_x == "in_more_x":
                                right_arr.append(right_pts)
                                right_pts = 0
                            flag_x = "in_less_x"
                            left_pts += 1
                            old_x = x

                    if flag == "in_more":
                        down_arr.append(down_pts)
                    elif flag == "in_less":
                        up_arr.append(up_pts)
                    if flag_x == "in_less_x":
                        left_arr.append(left_pts)
                    elif flag_x == "in_more_x":
                        right_arr.append(right_pts)

                    sum_x = (len(left_arr) + len(right_arr) + 1) // 2
                    sum_y = (len(up_arr) + len(down_arr) + 1) // 2

                    if sum_y in (4, 5, 6) and sum_x in (6, 8):
                        pred = ("*", 1)
                    elif sum_y == 4 and sum_x in (2, 3, 4):
                        pred = ("w", 1)
                    elif sum_y == 3 and sum_x in (3, 4):
                        pred = ("X", 1)
                    else:
                        pred = main(roi_pth)

                else:
                    pred = main(roi_pth)
            data_.append((filename, pred))
    return data_


def text_init(image, impth, col):
    H, W = cv2.imread("ocr/outputs/basic_input.jpg").shape[:2]
    df = extractROI(image)
    scale_img()
    print(colored('after scale', 'green'))
    pre = ""
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
    if col == "blue":
        txt_col = (255, 0, 0)
    elif col == "green":
        txt_col = (36, 255, 12)
    elif col == "red":
        txt_col = (0, 0, 255)
    elif col == "black":
        txt_col = (0, 0, 0)

    extimg = cv2.imread("ocr/outputs/basic_input.jpg")

    text_data = text_recognition(impth, col)
    ROI_number, count = 0, 0
    rec_text = ""
    rec_data = []
    for d in text_data:
        print(colored('in for loop', 'green'))
        for e, p, q, r, s, dot_x, dot_y in zip(
                df["ImageName"],
                df["x"],
                df["y"],
                df["x+w"],
                df["y+h"],
                df["dot_coordinate_x"],
                df["dot_coordinate_y"],
        ):
            print(colored(pre + e, "red"))
            print(colored(d[0], "green"))
            print(colored(d[0] == pre + e, "blue"))
            if d[0] == pre + e:
                word = d[1][0]
                confidence = d[1][1]
                if confidence == -1:
                    # draw shape
                    if word in ("rectangle", "square"):
                        text = cv2.rectangle(
                            extimg,
                            (p + d[1][2], q + d[1][3]),
                            (p + d[1][2] + d[1][4], q + d[1][3] + d[1][5]),
                            txt_col,
                            round(8 / 4032 * W),
                        )
                    elif word in ("circle", "oval"):
                        cnt_x, cnt_y = d[1][2], d[1][3]
                        cnt_w, cnt_h = d[1][4], d[1][5]
                        r = max(cnt_w / 2, cnt_h / 2)
                        text = cv2.circle(
                            extimg,
                            (int(p + cnt_x + cnt_w / 2),
                             int(q + cnt_y + cnt_h / 2)),
                            int(r),
                            txt_col,
                            round(8 / 4032 * W),
                        )
                    elif word in ("arrow", ):
                        text = cv2.arrowedLine(
                            extimg,
                            (p + d[1][2], q + d[1][3]),
                            (p + d[1][4], q + d[1][5]),
                            txt_col,
                            round(8 / 4032 * W),
                            tipLength=0.2,
                        )
                    rec_data.append(word)
                else:
                    print("WORD :: ", word)
                    char_set = "!@#$%^&*()_+=-,./<>?{}|\][ "
                    if col == "blue":
                        words = word.split('-')
                        checked_words = []
                        for w in words[1:]:
                            rec_text = spellchecker.SpellChecker().correction(w)
                            checked_words.append(rec_text)
                        word_next = '-'.join(checked_words)
                        rec_text = words[0]+'-'+word_next if word_next else words[0]
                        # ignore = [".B-waves", ".B-clouds", ".B-woves", "A-sun"]
                        # for i in char_set:
                        #     if word[0] == i:
                        #         words = word.split("-")
                        #         fixed_words = [words[0]]
                        #         for w in words[1:]:
                        #             checked_text = ""
                        #             if len(w) != 1:
                        #                 checked_text = (
                        #                     spellchecker.SpellChecker(
                        #                     ).correction(w[1:]))
                        #             rec_text = w[0] if w else '' + checked_text
                        #             # rec_text = w[0] if len(w) >= 1 else '' + checked_text
                        #             fixed_words.append(rec_text)
                        #         rec_text = "-".join(fixed_words)
                        #         break

                        #     else:
                        #         rec_text = "none"
                        #         checked_text = word
                        #         ROI = cv2.imread(f"ocr/resize/morph/{d[0]}")
                        #         gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                        #         _, thresh = cv2.threshold(
                        #             gray, 128, 255, cv2.THRESH_BINARY_INV)
                        #         cnts = cv2.findContours(
                        #             thresh, cv2.RETR_EXTERNAL,
                        #             cv2.CHAIN_APPROX_SIMPLE)
                        #         cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                        #         if len(word) != 1:
                        #             if len(cnts) != 1:
                        #                 words = word.split("-")
                        #                 print(colored(words, "green"))
                        #                 fixed_words = [words[0]]
                        #                 for w in words[1:]:
                        #                     fixed_words.append(
                        #                         spellchecker.SpellChecker(
                        #                         ).correction(w))
                        #                 checked_text = "-".join(fixed_words)
                        #             else:
                        #                 # checked_text = word[0]
                        #                 checked_text = word
                        #         rec_text = checked_text
                    if col == "red":
                        rec_text = word
                    rec_data.append(rec_text)
                    print("RECOGNISED DATA LIST : ", rec_data)

                    # Fixing the text position
                    fname = "_".join(
                        d[0].split("_")[1:]).split(".")[0] + ".jpg"
                    img = cv2.imread(f"ocr/ROI/{fname}")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 128, 255,
                                              cv2.THRESH_BINARY_INV)
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                    left_cnt = sorted(cnts,
                                      key=lambda x: cv2.boundingRect(x)[0])[0]
                    bottom_cnt = sorted(
                        cnts,
                        key=lambda x: cv2.boundingRect(x)[1] + cv2.
                        boundingRect(x)[3],
                    )[-1]

                    X = cv2.boundingRect(left_cnt)[0]
                    Y = (cv2.boundingRect(bottom_cnt)[1] +
                         cv2.boundingRect(bottom_cnt)[3])

                    # finding the width and height of the image to resize text
                    # based on it's dimension

                    if rec_text[0] == ".":
                        rec_text = rec_text[0] + rec_text[1:].strip()
                        if not dot_x and not dot_y:
                            text = cv2.putText(
                                extimg,
                                rec_text,
                                (p + X, q + Y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3 / 4032 * W,
                                txt_col,
                                round(8 / 4032 * W),
                            )
                        else:
                            text = cv2.putText(
                                extimg,
                                rec_text,
                                (
                                    int(dot_x - (5 * 3 / 4032 * W)),
                                    int(dot_y + 3 / 4032 * W),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3 / 4032 * W,
                                txt_col,
                                round(8 / 4032 * W),
                            )
                    else:
                        text = cv2.putText(
                            extimg,
                            rec_text,
                            (p + X, q + Y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3 / 4032 * W,
                            txt_col,
                            round(8 / 4032 * W),
                        )
                cv2.imwrite("ocr/outputs/basic_input.jpg".format(col), text)
                count += 1

    print("text_data info : ", text_data)
    return (df, rec_data, text_data)
