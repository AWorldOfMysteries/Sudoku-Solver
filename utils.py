# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:03:49 2023

@author: SAHIL
"""

import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                max_area = area
                biggest = approx
    return biggest, max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] 
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def initializePredictionModel():
    model = load_model('digitRecog.h5')
    return model

def getPrediction(boxes, model):
    result = []
    for image in boxes:
        ## PREPARE THE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img, (28, 28))
        img = img/255
        img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions) 
        # print(classIndex, probabilityValue)
        ## SAVE TO RESULT
        if probabilityValue > 0.6:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[0]/9) # = 50 here
    secH = int(img.shape[1]/9) # = 50 here
    for spanX in range(0,9):
        for spanY in range(0,9):
            if(numbers[(spanY*9)+spanX] != 0):
                cv2.putText(img, str(numbers[(spanY*9)+spanX]), 
                ((spanX*secW)+int(secW/2)-10, int((spanY+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img

#### SUDOKU SOLVING ALGORITHM
# We know that sudoku is a 9x9 2D matrix

def foundInCol(arr, col, num):  # Checks if num is aldready present in the column
    for i in range(9):
      if(arr[i][col] == num):
        return True
    return False


def foundInRow(arr, row, num):  # Checks if num is aldready present in the row
    for i in range(9):
      if(arr[row][i] == num):
        return True
    return False


def foundInBox(arr, row, col, num): # Checks if the num exists in the 3x3 grid
    startRow = row - (row % 3)
    startCol = col - (col % 3)
    for i in range(3):
      for j in range(3):
        if(arr[i + startRow][j + startCol] == num):
          return True
    return False

def isSafe(arr, row, col, num):
    return ((not foundInRow(arr, row, num)) and (not foundInCol(arr, col, num)) and (not foundInBox(arr, row, col, num)))

def foundEmptyCell(arr, loc): # Finds the location of the next empty cell 
    for i in range(9):
      for j in range(9):
        if(arr[i][j] == 0):
          loc[0] = i  # loc[0] will give the empty cell row
          loc[1] = j  # loc[1] will give the empty cell column
          return True
    return False

def solveSudoku(arr):

    l = [0,0]

    if(not foundEmptyCell(arr, l)): # Returns True when all spaces are filled by us
        return True

    row = l[0]  # Assigns the empty location
    col = l[1]  # got from the above function

    for num in range(1, 10):
        if(isSafe(arr, row, col, num)):
            arr[row][col] = num
            if(solveSudoku(arr)):
                return True
            arr[row][col] = 0       # If a num is safe, but there doesn't exist a solution with it; the location must be set to 0 for further iterations of num 

    return False  # Backtracking




# def stackImages(imgArray, scale):
#     rows = len(imgArray) # 2
#     cols = len(imgArray[0]) # 4
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0, rows):
#             for y in range(0, cols):
#                 imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2:
#                     imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_BGR2GRAY) 