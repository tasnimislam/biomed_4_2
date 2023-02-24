from PIL import Image
import cv2
import numpy as np
import math
import skimage
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion, square

def image_preprocessing(path):
    image = Image.open(path)
    image = np.array(image)

    # Preprocessing I have decided
    ad_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(ad_mean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_cann = cv2.threshold(otsu, 127, 255, cv2.THRESH_BINARY)
    # colored = cv2.cv2.cvtColor(bin_cann,cv2.COLOR_GRAY2RGB)
    # resized_image = cv2.resize(colored, (32, 32), interpolation=cv2.INTER_AREA)
    return bin_cann / 255.0



def computeAngle(block, minutiaeType):
    angle = 0
    (blkRows, blkCols) = np.shape(block);
    CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
    if (minutiaeType.lower() == 'termination'):
        sumVal = 0;
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle = -math.degrees(math.atan2(i - CenterY, j - CenterX))
                    sumVal += 1
                    if (sumVal > 1):
                        angle = float('nan');
        return (angle)
    elif (minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block);
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = []
        sumVal = 0;
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                    sumVal += 1
        if (sumVal != 3):
            angle = float('nan')
        return (angle)

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX;
        self.locY = locY;
        self.Orientation = Orientation;
        self.Type = Type;

def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif):
    FeaturesTerm = []

    minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2);
    RP = skimage.measure.regionprops(minutiaeTerm)

    WindowSize = 2
    FeaturesTerm = []
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = computeAngle(block, 'Termination')
        FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

    FeaturesBif = []
    minutiaeBif = skimage.measure.label(minutiaeBif, connectivity=2);
    RP = skimage.measure.regionprops(minutiaeBif)
    WindowSize = 1
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = computeAngle(block, 'Bifurcation')
        FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
    return (FeaturesTerm, FeaturesBif)


def ShowResults(skel, TermLabel, BifLabel):
    minutiaeBif = TermLabel * 0;
    minutiaeTerm = BifLabel * 0;

    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel;
    DispImg[:, :, 1] = skel;
    DispImg[:, :, 2] = skel;

    RP = skimage.measure.regionprops(BifLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1;
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1);
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0));

    RP = skimage.measure.regionprops(TermLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1;
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1);
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255));

    plt.figure(figsize=(6, 6))
    plt.title("Minutiae extraction results")
    plt.imshow(DispImg)
    plt.show()

def get_skel_mask(img):
    skel = skimage.morphology.skeletonize(img)
    skel = np.uint8(skel) * 255;
    mask = img * 255;

    return (skel, mask)


def getTerminationBifurcation(img, mask):
    img = img == 255;
    (rows, cols) = img.shape;
    minutiaeTerm = np.zeros(img.shape);
    minutiaeBif = np.zeros(img.shape);

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (img[i][j] == 1):
                block = img[i - 1:i + 2, j - 1:j + 2];
                block_val = np.sum(block);
                if (block_val == 2):
                    minutiaeTerm[i, j] = 1;
                elif (block_val == 4):
                    minutiaeBif[i, j] = 1;

    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(5))
    minutiaeTerm = np.uint8(mask) * minutiaeTerm
    return (minutiaeTerm, minutiaeBif)

def image_preprocessing_new(path):
    image = Image.open(path)
    image = np.array(image)

    # Preprocessing I have decided
    ad_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(ad_mean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_cann = cv2.threshold(otsu, 127, 255, cv2.THRESH_BINARY)
    bin_cann = bin_cann/255.0
    # colored = cv2.cv2.cvtColor(bin_cann,cv2.COLOR_GRAY2RGB)
    # resized_image = cv2.resize(colored, (32, 32), interpolation=cv2.INTER_AREA)

    skel, mask = get_skel_mask(bin_cann)
    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);
    FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif)
    BifLabel = skimage.measure.label(minutiaeBif, connectivity=1);
    TermLabel = skimage.measure.label(minutiaeTerm, connectivity=1);

    return BifLabel / 255.0, TermLabel / 255.0

def image_preprocessing_final(path):
    image = Image.open(path)
    image = np.array(image)

    # Preprocessing I have decided
    ad_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(ad_mean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_cann = cv2.threshold(otsu, 127, 255, cv2.THRESH_BINARY)
    # colored = cv2.cvtColor(bin_cann,cv2.COLOR_GRAY2RGB)
    colored = bin_cann
    colored = colored[38:268,20:236]
    # colored=fingerprint_enhancer.enhance_Fingerprint(colored)
    # colored = cv2.cvtColor(bin_cann,cv2.COLOR_GRAY2RGB)
    colored = cv2.cvtColor(colored,cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(colored, (256,256), interpolation=cv2.INTER_AREA)
    return resized_image / 255.0