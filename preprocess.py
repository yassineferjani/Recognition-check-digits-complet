import cv2
import numpy as np


def money_region(image, height_prop=0.2, width_prop=0.7):
    '''
        returns the region that contains money amount

        params:
            * image : GrayScale image
            * height_prop : the height of the portion that may contain the money amount
            * width_prop : the width of the portion that may contain the money amount

        the default values are specified based on the Tunisian's bank cheque format (upper right region)
    '''

    (h, w) = image.shape
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh[: int(h * height_prop), int(w * width_prop): int(w * 0.97)]


def preprocess(gray_image):
    '''
        returns binary image after applying median Blur

        params:
            *gray_image: GrayScale image
    '''

    blur = cv2.medianBlur(gray_image, 3)  # Median Blur to maintain image quality

    kernel = np.ones((2, 2))
    blur = cv2.dilate(blur, kernel)
    # dil=cv2.erode(blur, kernel)
    erosion = cv2.ximgproc.thinning(blur)

    return erosion


def segmentation(image, mode):
    """

    :param image: binary image
    :param mode: 1 for the horizontal segmentation and 0 for the vertical segmentation
    :return: Sequences of the regions containing information

    """
    if mode == 0:

        pix_density = []
        (h, w) = image.shape

        for idx in range(w):
            col = image[0: h, idx: idx + 1]
            pix_density.append(np.sum(col))

    if mode == 1:
        pix_density = np.sum(image, axis=1, keepdims=True)

    x = list(np.nonzero(pix_density)[0])
    seq = [x[0]]
    seqs = []

    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1:
            seq.append(x[i - 1])
            seqs.append(seq)
            seq = []
            seq.append(x[i])
        if i == (len(x) - 1):
            seq.append(x[i])
            seqs.append(seq)

    return seqs


def horizontal_segmentimage1(imgorig, image):
    """

    :param imgorig: The original image
    :param image: The binary image
    :return: return the largest region horizontally of the original image
    """
    segs = segmentation(image, 1)
    max = 0
    idx = 0

    for i in range(len(segs)):
        if segs[i][1] - segs[i][0] > max:
            max = segs[i][1] - segs[i][0]
            idx = i

    return imgorig[segs[idx][0] - 3: segs[idx][1], :]


def horizontal_segment(image):
    """

    :param image: The binary image
    :return: return the largest region horizontally of the binary image
    """
    segs = segmentation(image, 1)
    max = 0
    idx = 0

    for i in range(len(segs)):
        if segs[i][1] - segs[i][0] > max:
            max = segs[i][1] - segs[i][0]
            idx = i

    return image[segs[idx][0]: segs[idx][1], :], [segs[idx][0], segs[idx][1]]


def vertical_segmentation(image):
    """

    :param image: The binary image
    :return: returns vertical regions of the binary image
    """
    segs = segmentation(image, 0)

    chiffres = []

    for i in range(len(segs)):
        if (segs[i][1] - segs[i][0]) > 1:
            chiffres.append([segs[i][0], segs[i][1]])

    return chiffres