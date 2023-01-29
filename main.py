# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:05:00 2023

@author: SAHIL
"""

# Imports ---------------------------------------
from utils import *
# -----------------------------------------------

################################################
pathImage = r"Images/sudoku_img.jpeg"
heightImage = 450
widthImage = 450
model = initializePredictionModel()

##################################################

## STEP 1 : READING AND PREPROCESSING IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (heightImage, widthImage))
imgBlank = np.zeros((heightImage, widthImage, 3), np.uint8)   
imgThresholded = preProcess(img)


## STEP 2 : FIND ALL CONTOURS
imgContours = img.copy()
imgBigContour = img.copy()
contours, heirarchy = cv2.findContours(imgThresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)


## STEP 3 : FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImage, heightImage))
    imgWarpGrayed = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgDetectedDigits = imgBlank.copy()


## STEP 4 : SPLIT THE WARPED SUDOKU IMAGE AND PREDICT THE DIGITS
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpGrayed)
# print(boxes[0].shape) := (50, 50)
numbers = getPrediction(boxes, model)
# print(numbers)
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
numbers = np.asarray(numbers)
posList = np.where(numbers > 0, 0, 1) # Set bit at blank spaces 


#### STEP 5 : SOLVE
board = np.reshape(numbers, (9, 9))
# print(board)
if(solveSudoku(board)):
    solvedBoard = board
else:
    print("Error")
# print(solvedBoard)
solvedBoardFlattened = []
for subarr in solvedBoard:
    for item in subarr:
        solvedBoardFlattened.append(item)

solvedNumbers = solvedBoardFlattened*posList
imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers, (0, 255, 0))

#### STEP 6 : OVERLAYING THE SOLUTION
pts1 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
pts2 = np.float32(biggest)

imgInvWarped = img.copy()
invMatrix = cv2.getPerspectiveTransform(pts1, pts2)
imgInvWarped = cv2.warpPerspective(imgSolvedDigits, invMatrix, (widthImage, heightImage))
imgOverlayed = cv2.addWeighted(imgInvWarped, 1, img, 0.4, 1)

cv2.imshow('OUTPUT', imgOverlayed)
cv2.imshow('INPUT IMAGE', img)
# cv2.imshow('Images2', imgSolvedDigits)
# cv2.imshow('Images3', )
cv2.waitKey(0)



















