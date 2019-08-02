#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: Allen(Zifeng) An
@course: 
@contact: anz8@mcmaster.ca
@file: image_detection.py
@time: 8/2/2019 3:17 PM
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# takes in an image and returns the canny edge image
# gray scale --> noise reduction  --> canny edge detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# in gray scale picture: 0 --> black, 255 -->white
def region_of_interest(image): # this image is canny --> black and white
    height=image.shape[0] # height of the image 480
    width=image.shape[1] #640

    # mask (bitwise AND) image --> region of interest
    # our mask is a polygon, the origin is the top left corner of the window
    triangle = np.array([
        [(0,300),(0,140),(250,0),(450,0),(width,140),(width,300)]])

    mask = np.zeros_like(image) # all 0 array --> black
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image,mask)
    # you can use imshow function to show any of these images
    return masked_image

#put lines on a black image
def display_lines(image,lines):
    line_image=np.zeros_like(image) #black background
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            #img, point1, point2, rbg colour, thickness
        return line_image

def average_slope_intercept_left(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4) # line here is numpy ndarray
        parameters= np.polyfit((x1,x2),(y1,y2),1) # returns slope and y-intercept of each line
        slope, intercept=parameters[0],parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    if len(left_fit) == 0: # np.average will throw an error if fit is empty
        raise Exception("Left Empty.")

    left_fit_avg=np.average(left_fit,axis=0)
    left_line= make_coordinates(image,left_fit_avg)
    return np.array([left_line])

def average_slope_intercept_right(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters= np.polyfit((x1,x2),(y1,y2),1)
        slope, intercept=parameters[0],parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    if len(right_fit) == 0:
        raise Exception("Right Empty.")

    right_fit_avg = np.average(right_fit, axis=0)
    right_line=make_coordinates(image,right_fit_avg)
    return np.array([right_line])

# this function is never used
def average_slope_intercept(image,lines):
    return np.concatenate((average_slope_intercept_left(image,lines),average_slope_intercept_right(image,lines)))


def make_coordinates(image, line_parameters): #takes the line parameters, returns the coordinates
    # print("in make_coordinates(): ", line_parameters)
    slope,intercept= line_parameters
    # y1=image.shape[0]
    # y2=int(y1*0.6) # 60 percent of the screen from bottom up
    y1 = 300 # arbitrarily defined
    y2 = 50
    x1=int((y1-intercept)/slope) # linear equation of the line
    x2 = int((y2 - intercept) / slope)

    return np.array([x1,y1,x2,y2])

def calculate_angle(car,centre):
    #takes the coordinates of the car and the current centre of the lane
    rise, run =centre[1]-car[1],centre[0]-car[0]
    if run == 0:
        print('Going straight')
    else:
        slope = rise/run
        radian = math.atan(math.fabs(slope))
        angle = 90-math.degrees(radian)
        if angle<4: print('Going straight') # noise control, ignore 4 degrees or less deviation
        else:
            if slope <0: print('Turning right '+str(round(angle,2))+' degrees')
            else: print('Turning left '+str(round(angle,2))+' degrees')


def mid_point(line): # return the mid point of the given line
    x1,y1,x2,y2=line.reshape(4)
    #640 is the width of the image
    slope = (y2-y1)/(x2-x1)
    intercept= y1-slope*x1
    if slope <0: margin_point= 0,intercept
    else: margin_point= 640, slope*640+intercept
    mid_x=(margin_point[0]+x2)/2
    mid_y=(margin_point[1]+y2)/2
    return mid_x,mid_y



# img = cv2.imread('arc.png')
img = cv2.imread('opencv_frame_1.png')

lane_image = np.copy(img)
canny_image = canny(lane_image)
cropped_img = region_of_interest(canny_image)
# HoughLineP(img,pixel quantity, theta, number of intersections in the voting box)
# transform into
lines = cv2.HoughLinesP(cropped_img, 1, np.pi / 180, 40, np.array([]), minLineLength=4,
                        maxLineGap=10)  # minLineLength=40,maxLineGap=5

averaged_lines_right = average_slope_intercept_right(lane_image, lines)
line_image= display_lines(lane_image, lines)  # black background with blue lanes
    # avg_line_image_right = display_lines(lane_image, averaged_lines_right)
# cv2.imshow('result',line_image)
cv2.waitKey(0)


right_flag,left_flag= False, False
try:
    averaged_lines_right = average_slope_intercept_right(lane_image, lines)
    # line_image = display_lines(lane_image, lines)  # black background with blue lanes
    avg_line_image_right = display_lines(lane_image, averaged_lines_right)
    # cv2.imshow('right',avg_line_image_right)
# cv2.imshow('canny',canny_image)
# cv2.imshow('result',cropped_img)
# cv2.imshow('lines',line_image)
# cv2.imshow('avg lines',avg_line_image)
except Exception as e:
    print('Right Error!',e)
    right_flag=True


try:
    averaged_lines_left = average_slope_intercept_left(lane_image, lines)
    avg_line_image_left = display_lines(lane_image, averaged_lines_left)
    # cv2.imshow('left',avg_line_image_left)
except Exception as e:
    print('Left Error!', e)
    left_flag = True

if left_flag ==False and right_flag == False:
    # avg=np.concatenate((avg_line_image_left,avg_line_image_right))
    combo_image = cv2.addWeighted(lane_image, 0.8, avg_line_image_left, 1, 1)
    combo_image = cv2.addWeighted(combo_image, 1, avg_line_image_right, 1, 1)

    mid_right = tuple(int(x) for x in mid_point(averaged_lines_right))
    mid_left = tuple(int(x) for x in mid_point(averaged_lines_left))
    cv2.circle(combo_image, mid_right, 5, (0,0,255),thickness=5)
    cv2.circle(combo_image, mid_left, 5, (0, 0, 255), thickness=5)
    centre= int((mid_left[0]+mid_right[0])/2),int((mid_left[1]+mid_right[1])/2)
    cv2.circle(combo_image, centre, 5, (192, 0, 192), thickness=5)

    # print('right',mid_point(averaged_lines_right))
    # print('left',mid_point(averaged_lines_left))
    # print('right',mid_right)
    # print('left', mid_left)
    cv2.imshow('combo', combo_image)
    cv2.waitKey(0)
    # plt.imshow(combo_image) #shows coordinates
    # plt.show()

elif left_flag == False and right_flag == True:
    combo_image = cv2.addWeighted(lane_image, 0.8, avg_line_image_left, 1, 1)
    mid_left = tuple(int(x) for x in mid_point(averaged_lines_left))
    cv2.circle(combo_image, mid_left, 5, (0, 0, 255), thickness=5)
    centre = mid_left[0] +500, mid_left[1]
    cv2.circle(combo_image, centre, 5, (192, 0, 192), thickness=5)
    cv2.imshow('combo', combo_image)
    cv2.waitKey(0)
elif left_flag == True and right_flag == False:
    combo_image = cv2.addWeighted(lane_image, 0.8, avg_line_image_right, 1, 1)
    mid_right = tuple(int(x) for x in mid_point(averaged_lines_right))
    cv2.circle(combo_image, mid_right, 5, (0, 0, 255), thickness=5)
    centre = mid_right[0] -500, mid_right[1]
    cv2.circle(combo_image, centre, 5, (192, 0, 192), thickness=5)
    cv2.imshow('combo', combo_image)
    cv2.waitKey(0)
else:
    cv2.imshow('combo', lane_image)
    cv2.waitKey(0)

    plt.imshow(canny_image) #shows coordinates
    plt.show()

