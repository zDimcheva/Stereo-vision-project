import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

numDisparities = 64
blockSize = 5
edgesL = 0
edgesR = 0

# ================================================ #
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    return disparity  # floating point image
# ================================================

baseline = 174.019
doffs = 114.291
focal = 5806.559

# ================================================ #
def plot(disparity, baseline, focal, doffs):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    X_w = []
    Y_w = []
    Z_w = []

    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            z = baseline * focal / (disparity[i, j] + doffs)
            x = z * i / focal
            y = z * j / focal

            if (z < 7500):
                X_w += [x]
                Y_w += [y]
                Z_w += [z]

    # Plot depths
    ax = plt.axes(projection='3d')
    ax.scatter(X_w, Y_w, Z_w, s=1, c='green')

    # Labels
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    ax.view_init(90, 0)

    plt.savefig('myplot.pdf', bbox_inches='tight')  # Can also specify an image, e.g. myplot.png
    plt.show()

def numD(x):
    global numDisparities
    numDisparities = x * 16
    print(numDisparities)
    print(blockSize)
    disparity = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)

def blockS(y):
    global blockSize
    blockSize = (y + 2) * 2 + 1
    print(numDisparities)
    print(blockSize)
    disparity = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)

# ================================================ #
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgLa = cv2.imread(filename, flags=0)
    imgL = cv2.resize(imgLa, (740, 505))
    img_blur = cv2.GaussianBlur(imgL, (3, 3), sigmaX=0, sigmaY=0)
    edgesL = cv2.Canny(image=img_blur, threshold1=50, threshold2=120)

    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'umbrellaR.png'
    imgRa = cv2.imread(filename, flags=0)
    imgR = cv2.resize(imgRa, (740, 505))
    img_blur = cv2.GaussianBlur(imgR, (3, 3), sigmaX=0, sigmaY=0)
    edgesR = cv2.Canny(image=img_blur, threshold1=50, threshold2=125)

    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Disparity A', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity A', imgL)

    cv2.namedWindow('Disparity B', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity B', imgR)

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Create TrackBar
    cv2.createTrackbar('numDisparities', 'Disparity', 0, 40, numD)  # Create TrackBar
    cv2.createTrackbar('blockSize', 'Disparity', 0, 20, blockS)

    # Get disparity map
    disparity = getDisparityMap(edgesL, edgesR, 64, 5)

    # Normalize for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    # plot(disparity, baseline, focal, doffs)

    # Wait for spacebar press or escape before closing,
    # otherwise the window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
    cv2.destroyAllWindows()
