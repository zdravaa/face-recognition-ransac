# -*- coding: utf-8 -*-
"""
Created on Sat Sept  15 11:39:04 2022

@author: User
"""

import cv2 as cv
import sys
import time
# globbing utility
import glob
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path


img_counter = 0
cam = cv.VideoCapture(0)
lista = [0]
bestMatch = 10

while True:
    #read values from camera
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("Take a picture for query", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # if we press "ESC", camera closes
        print("Escape hit, closing camera...")
        break
    elif k%256 == 32:
        # if we press "SPACE", we take a picture
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv.destroyAllWindows()


#read the picture, detect face in it
image = cv.imread('opencv_frame_0.png') 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces!".format(len(faces)))

#crop the face from taken picture 
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceDet = image[y:y + h, x:x + w]
    #cropped_image = image[80:280, 150:330]
    cv.imwrite("Cropped Image.jpg", faceDet)


status = cv.imwrite('faceDetected.jpg', faceDet)
print("[INFO] Image faceDetected.jpg written to filesystem: ", status)



cv.destroyAllWindows() 


# read all images from imageFolder

path = "imageFolder/*/*.*"
for file in glob.glob(path):
    image_read = cv.imread(file)
    # conversion numpy array into rgb image to show
    c = cv.cvtColor(image_read, cv.COLOR_BGR2RGB)
    #cv.imshow('Color image', c)
    # wait for 1 second
    #k = cv.waitKey(1000)
    # destroy the window
    #cv.destroyAllWindows()
  

    queryImagePath = "Cropped Image.jpg"
    MIN_MATCH_COUNT = 10
    imgQuery = cv.imread(queryImagePath) # query image
    img1 = cv.cvtColor(imgQuery, cv.COLOR_BGR2GRAY) # convert image to gray 
    img2 = cv.imread(file) #pic from folder
  
    #initialize SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test
    good = []
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
            if len(good)>MIN_MATCH_COUNT:
                lista.append(len(good))
                if len(good) > bestMatch:
                    bestMatch = len(good)
                    bestImageMatch = os.path.split(file)[-1]
                    fullImagePath = os.path.split(file)[-2]
                    #fullPathNew = os.path.basename(file) ovo daje ime file-a, koristiti umjesto ovog gore
                    fullPathNew = os.path.abspath(file)
                    #fullPathNew = Path(file)
                    
                    
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv.perspectiveTransform(pts,M)
                img2 = cv.polylines(img2,[np.int32(dst_pts)],True,255,3, cv.LINE_AA)
                print("Matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                print("this is the best match - ", bestImageMatch, "with ", bestMatch, "matches")
                print("This is the full path: ", fullPathNew)

                
            else:
                print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                matchesMask = None
                
                
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
#img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#plt.imshow(img3, 'gray'),plt.show()



#if bestImageMatch != "":
    
 #   window_name = "Best match"
  #  bestMatchPath = 'imageFolder/*/' + bestImageMatch
   # showBestMatch = cv.imread(bestMatchPath)
    #cv.imshow(window_name, showBestMatch)
if bestImageMatch != '':
    print("What is the best image  match? - ", bestImageMatch, "with ", bestMatch, "matches")
    
    imgResultPath = fullImagePath.replace("\\", "/" )
    print(imgResultPath)
    imageWithoutExtension = os.path.splitext(bestImageMatch)[0]
    imgResultPath = 'imageFolder/zdravko/' + "/" + bestImageMatch
    matchFoundText = "Match found, hello " + imageWithoutExtension
    imgResult = cv.imread(imgResultPath)
    plt.text(1, 5, matchFoundText, fontsize = 16)
    plt.imshow(imgResult),plt.show()
else:
    print("Ne postoji match")


cv.destroyAllWindows()   



# Ransac
#https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

#Prolazak kroz folder slika
#https://sanpreetai.medium.com/read-multiple-images-from-a-folder-using-python-cv2-725f0b0447d4