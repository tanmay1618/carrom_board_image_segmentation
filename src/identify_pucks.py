import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys


def filter_contours(contours,min_radius,max_radius):
    radius_data = []
    area_data = []
    final_contours = []
    for index,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        if radius > min_radius and radius < max_radius and area >= 3.14*min_radius**2 and area <= 3.14*max_radius**2:
            area_data.append(area)
            radius_data.append(radius)
            final_contours.append(cnt)
    return final_contours

def make_contours(threshed):
    _,contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(img,contours):
    img_test = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 3)
    plt.imshow(img_test)
    plt.show()

def preprocess_image(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return gray	
	
def identify_black_pucks(img):
    th, threshed = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    return(filter_contours(make_contours(threshed),10,50))
    
def identify_white_pucks(img):
    th, threshed = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    return(filter_contours(make_contours(threshed),10,50))

def identify_striker(img):
    th, threshed = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    return(filter_contours(make_contours(threshed),30,50))

def find_pucks(location):
    img = Image.open(location)
    cv_img = np.array(img) 
    processed_img = preprocess_image(cv_img)
    black_puck_contours = identify_black_pucks(processed_img)
    print(len(black_puck_contours))
    white_puck_contours = identify_white_pucks(processed_img)
    print(len(white_puck_contours))
    striker_contour = identify_striker(processed_img)
    print(len(striker_contour))
    drawContours(cv_img,black_puck_contours+white_puck_contours+striker_contour)

        
if __name__ ==  "__main__":
    #To do: Crop the carrom board better and automatically
    #To do: Improve puck identification
    #To do: Add video data
    location = sys.argv[1] #"../data/crop.jpg"
    find_pucks(location)

	



