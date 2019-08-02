#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: Allen(Zifeng) An
@course: 
@contact: anz8@mcmaster.ca
@file: detect.py
@time: 7/30/2019 11:50 AM
'''
#This is the example with minor modification from the youtube tutorial
# https://www.youtube.com/watch?v=eLTLtUVuuy4

import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height=image.shape[0]
    triangle= np.array([
        [(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)

            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        return line_image

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters= np.polyfit((x1,x2),(y1,y2),1) # returns slope and y-intercept of each line
        slope, intercept=parameters[0],parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    if len(left_fit) == 0 or len(right_fit) == 0:
        raise Exception("Empty left fit, should continue.")

    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line= make_coordinates(image,left_fit_avg)
    right_line=make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])

def make_coordinates(image, line_parameters):
    print("in make_coordinates(): ", line_parameters)
    slope,intercept= line_parameters
    y1=image.shape[0]
    y2=int(y1*0.6) # 60 percent of the screen from bottom up
    x1=int((y1-intercept)/slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1,y1,x2,y2])

# img = cv2.imread('test_image.jpg')
# lane_image=np.copy(img)
# canny_image=canny(lane_image)
# cropped_img=region_of_interest(canny_image)
# #HoughLineP(img,pixel quantity, theta, number of intersections in the voting box)
# #transform into
# lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_lines(lane_image,averaged_lines) #black background with blue lanes
# combo_image= cv2.addWeighted(lane_image,0.8,line_image,1,1)



# cv2.imshow('result',combo_image)
# cv2.waitKey(0)

# plt.imshow(canny) #shows coordinates
# plt.show()

cap= cv2.VideoCapture('test2.mp4')

while (cap.isOpened()) == True:
    _,frame =cap.read()
    if  _ == False:
        break
    canny_image = canny(frame)
    cropped_img = region_of_interest(canny_image)
    # HoughLineP(img,pixel quantity, theta, number of intersections in the voting box)
    # transform into
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    print("from hough lines: ", lines.shape, lines)

    try:
        averaged_lines = average_slope_intercept(frame, lines)
    except Exception as e:
        print("Error!", e)
    line_image = display_lines(frame, averaged_lines)  # black background with blue lanes
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    cv2.waitKey(1)

