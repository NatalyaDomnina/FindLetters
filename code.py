import cv2
import os
import numpy as np
from scipy.spatial import distance
import random

def filters(img):
    black_bool = True
    white_bool = False
	
    G = img.copy()
    gpA = [G]
    for i in range(4):
        G = cv2.pyrDown(G)
        gpA.append(G)
		
    img = gpA[2]
    h, w = img.shape

    filter_0 = np.ones((h, w), bool)
	
    filter_1 = np.ones((h, w), bool)
    filter_1[0:h, 0:int(w / 2)] = white_bool

    filter_2 = np.ones((h, w), bool)
    filter_2[int(h / 2):h, 0:w] = white_bool

    filter_3 = np.logical_not(np.logical_xor(filter_1, filter_2))

    filter_4 = np.ones((h, w), bool)
    filter_4[0:h, 0:int(w / 4)] = white_bool
    filter_4[0:h, int(3 * w / 4):w] = white_bool

    filter_6 = np.logical_not(np.logical_xor(filter_4, filter_2))

    filter_9 = np.ones((h, w), bool)
    filter_9[0:int(h / 4), 0:w] = white_bool
    filter_9[int(3 * h / 4):h, 0:h] = white_bool

    filter_7 = np.logical_not(np.logical_xor(filter_1, filter_9))

    filter_8 = np.logical_not(np.logical_xor(filter_4, filter_9))

    filter_5 = np.ones((h, w), bool)
    filter_5[0:h, int(w / 4):int(2 * w / 4)] = white_bool
    filter_5[0:h, int(3 * w / 4):w] = white_bool

    filter_10 = np.ones((h, w), bool)
    filter_10[0:int(h / 4), 0:w] = white_bool
    filter_10[int(2 * h / 4):int(3 * h / 4), 0:w] = white_bool

    filter_11 = np.logical_not(np.logical_xor(filter_2, filter_5))

    filter_12 = np.logical_not(np.logical_xor(filter_1, filter_10))

    filter_13 = np.logical_not(np.logical_xor(filter_5, filter_9))

    filter_14 = np.logical_not(np.logical_xor(filter_4, filter_10))

    filter_15 = np.logical_not(np.logical_xor(filter_5, filter_10))

    filters = [filter_0, filter_1, filter_2, filter_3, filter_4, filter_5,
               filter_6, filter_7, filter_8, filter_9, filter_10, filter_11,
               filter_12, filter_13, filter_14, filter_15]
    
	filtet_number = 0
    full_res = []
    for filter in filters:
        res = 0
        for i in range(h):
            for j in range(w):
                if filter[i][j] == black_bool:
                    res += img[i][j]
                elif filter[i][j] == white_bool:
                    res -= img[i][j]
        #print('filter ' + str(filtet_number) + ' = ' + str(res))
        filtet_number += 1
        full_res.append(res)

    #print('full_res = ' + str(full_res))
    return full_res

def create_sign():
    images = os.listdir('C:/Users/Natali/Documents/letters_256')
    print (images)
    for img in images:
        folder = 'C:/Users/Natali/Documents/letters_256/'
        img_input = cv2.imread(os.path.join(folder,img))
        print(os.path.join(folder,img))
        h, w, c = img_input.shape
        f = open('sign2lvl40.txt', 'a')
        img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        res_no_grid = filters(img_input_gray)
        for index in res_no_grid:
            f.write(str(index))
            f.write(' ')
        f.write('\n')
        f.close()

def sign_to_list():
    f = open('sign2lvl40.txt', 'r')
    sign = []
    line = f.readline()
    sign.append(line.rstrip())
    while line:
        line = f.readline()
        sign.append(line.rstrip())
    sign = [i.split(' ') for i in sign]
    sign = sign[:-1]
    for i in range(len(sign)):
        sign[i] = [int(n) for n in sign[i]]
    return sign

def sp_noise(image,prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#create_sign()
sign = sign_to_list()
im = cv2.imread('C:/Users/Natali/Documents/test8-2.png')
im = sp_noise(im, 0)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray_blur, 160, 255, cv2.THRESH_BINARY_INV)
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
count_letter = 0

for rect in rects:
    if (rect[3] < 20 or rect[2] < 20):
        continue
    im2 = im_gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    im2 = cv2.equalizeHist(im2)
    im2 = cv2.medianBlur(im2, 3)
    im2 = cv2.resize(im2, (40, 40), cv2.INTER_AREA)

    res = filters(im2)
    res = np.asarray(res)
    res_coef = []
    for i in range(len(sign)):
        sign[i] = np.asarray(sign[i])
        res_coef.append(distance.euclidean(sign[i], res))

    letters = ['A', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
               'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
               'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
               'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й',
               'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
               'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
    ind_min = np.argmin(res_coef)

    if letters[ind_min] == 'Д' or letters[ind_min] == 'д':
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
        count_letter += 1

cv2.imshow('123', im)
print(count_letter)

cv2.waitKey(0)
