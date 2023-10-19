import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d 
from matplotlib import pyplot as plt

numDisparities = 64
blockSize = 5
edgesL = 0
edgesR = 0
imgL = 0
imgR = 0
k = 0

# ================================================ #
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    return disparity  # floating point image
# ================================================

baseline = 174.019
doffs = 114.291
focal = 5806.559

def numD(x):
    global numDisparities
    numDisparities = x * 16

    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)

    depth_fun(k)

def blockS(y):
    global blockSize
    blockSize = (y + 2) * 2 + 1

    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)

    depth_fun(k)

def depth_fun(z):
    global depth, k
    k = z
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)

    depth = [[0]*len(disparity[0]) for _ in range(len(disparity))]
    depth = 1 / (disparity + k)
    depth = np.array(depth)

    depthImg = np.interp(depth, (depth.min(), depth.max()), (0, 255))
    result = cv2.GaussianBlur(imgL, (41, 41), 0)

    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            if depthImg[i, j] < k:
                result[i, j] = imgL[i, j]

    cv2.imshow('Disparity', result)

# ================================================
if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, flags=0)
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'girlR.png'
    imgR = cv2.imread(filename, flags=0)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Disparity A', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity A', imgL)

    cv2.namedWindow('Disparity B', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity B', imgR)

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    # Create TrackBar
    cv2.createTrackbar('numDisparities', 'Disparity', 0, 40, numD)
    cv2.createTrackbar('blockSize', 'Disparity', 0, 20, blockS)
    cv2.createTrackbar('k', 'Disparity', 0, 400, depth_fun)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 16, 5)

    # Normalize for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Wait for spacebar press or escape before closing,
    # otherwise the window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
    cv2.destroyAllWindows()
