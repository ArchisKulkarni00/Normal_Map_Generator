import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

readfiles = os.listdir(os.getcwd())
files = [file for file in readfiles if file[-4:] in ['.png','.PNG','.jpg','.JPG']]
power=7
alpha = 2
beta = -20
for _ in files:
    img=cv2.imread(_)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalised =cv2.bitwise_not(cv2.equalizeHist(gray))
    b_c = cv2.addWeighted(equalised,alpha,equalised,0,beta)
    r = cv2.normalize(cv2.Sobel(b_c,cv2.CV_64F,1,0,ksize=5),None,0,255,cv2.NORM_MINMAX)
    g = cv2.normalize(cv2.Sobel(b_c,cv2.CV_64F,0,1,ksize=5),None,0,255,cv2.NORM_MINMAX)
    b = (power*255)/(np.ones((img.shape[0],img.shape[1]),dtype='float64'))
    map1 = cv2.merge((b,g,r))
    cv2.imwrite(_[slice(0,-4)]+'_Normal_Map.tiff',map1)
    cv2.imwrite(_[slice(0,-4)]+'_Height_Map.tiff',b_c)
cv2.destroyAllWindows()