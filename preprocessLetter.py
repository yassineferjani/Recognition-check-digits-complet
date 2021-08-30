import cv2
import numpy as np
import math
def ecrit_reg(img, x1 = 0.15, x2 = 0.55):
  if len(img.shape)==2:
    h, _ = img.shape
  else:
    h, _,_ = img.shape

  return img[int(h*x1): int(h*x2), :]


def detectionLigne(image):
    result = image.copy()
    thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)[1]

    kernel = np.ones((1, 3), np.uint8)

    thresh = cv2.dilate(thresh, kernel)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=5)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    dtype = [('horizontal', int), ('left', int), ('right', int)]
    ligne = []
    for i in range(len(cnts)):
        x = []
        y = []
        for j in range(len(cnts[i])):
            x.append(cnts[i][j][0][0])
            y.append(cnts[i][j][0][1])
        ligne.append((np.min(y), min(x), max(x)))

    a = np.array(ligne, dtype=dtype)
    ligne = np.sort(a, order='horizontal')
    ligne = list(ligne)

    lines = []

    while (len(ligne) != 0):

        if not lines:
            lines.append(ligne[-1])

        else:
            line = ligne.pop()

            if (lines[-1][0] - line[0]) < 20:
                lines[-1][0] = min(lines[-1][0], line[0])
                lines[-1][1] = min(lines[-1][1], line[1])
                lines[-1][2] = max(lines[-1][2], line[2])

            else:
                lines.append(line)

    ########
    lines.reverse()
    lines_by_distance = [((line[2] - line[1]), line) for line in lines]
    lines_by_distance.sort()
    top3 = [line[-1] for line in lines_by_distance[-3:]]

    a = np.array(top3, dtype=dtype)
    threelines = np.sort(a, order='horizontal')

    return threelines

def extracthandwritten (image,ligne):
  reg1= cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)[1]

  mask = reg1.copy()
  y,x1,x2 = ligne[0]
  mask[:y,:x1]=np.zeros_like(mask[:y,:x1])
  mask[:y,x2:]=np.zeros_like(mask[:y,x2:])
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
  img= cv2.erode(mask, kernel)
  return mask


def traitementLigne(img):
    kernel = np.ones((3, 3))

    im = img.copy()

    # im = cv2.dilate(img, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        if cv2.contourArea(c) < 20:
            im[y: (y + h), x: (x + w)] = np.zeros_like(im[y: (y + h), x: (x + w)])
    return im


def extractionLigne(image, ligne):
    img1 = image[: ligne[0][0] + 10, ligne[0][1]: ligne[0][2]]
    img2 = image[ligne[0][0]: ligne[1][0] + 10, ligne[1][1]: ligne[1][2]]
    img3 = image[ligne[1][0]: ligne[2][0] + 10, ligne[2][1]: ligne[2][2]]

    return [img1, img2, img3]


def segmentationLigne(imageligne):
    mots = vertical_segmentation(imageligne)
    lines = []

    while (len(mots) != 0):

        if not lines:
            lines.append(mots[-1])

        else:
            line = mots.pop()

            if (lines[-1][0] - line[-1]) < 10:
                lines[-1][0] = min(lines[-1][0], line[0])
                lines[-1][1] = max(lines[-1][1], line[1])

            else:
                lines.append(line)
    lines.reverse()

    return lines

def denosingWord(imageligne,listeMots):
  mots_25 = []
  for i in range((len(listeMots))):
    kernel=np.ones((2,2))
    img=cv2.erode(imageligne[:, listeMots[i][0]: listeMots[i][1]],kernel)
    contour_letter, _= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contour_letter:
      rect = cv2.boundingRect(c)
      x, y, w, h = rect
      if cv2.contourArea(c) < 5:
        img[y: (y+h),  x: (x+w)] = np.zeros_like( img[y: (y+h),  x: (x+w)])
    mots_25.append(img)
  return mots_25


def segmentationWord(imagemot):
    image = cv2.ximgproc.thinning(imagemot)
    image = image // 255

    pix_density = []
    (h, w) = image.shape

    for idx in range(w):
        col = image[0: h, idx: idx + 1]

        pix_density.append(np.sum(col))

    x = []
    seq = []
    seqs = []

    for i in range(len(pix_density)):
        if pix_density[i] == 1:
            x.append(i)

    if len(x) == 0:
        return [0, w]

    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1:
            seq.append(x[i - 1])
            seqs.append(seq)
            seq = []
            seq.append(x[i])

        if i == (len(x) - 1):
            seq.append(x[i])
            seqs.append(seq)

    lens = [i[-1] - i[0] for i in seqs]

    x = round(np.mean(lens))
    lens1 = []

    for i in range(len(lens)):
        if lens[i] >= x:
            lens1.append(seqs[i])

    seg = [round((i[-1] + i[0]) / 2) for i in lens1]
    seg.insert(0, 0)
    seg.append(len(pix_density))

    return seg


def segmentation(image, mode):
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


def horizontal_segment(image):
    segs = segmentation(image, 1)
    max = 0
    idx = 0

    for i in range(len(segs)):
        if segs[i][1] - segs[i][0] > max:
            max = segs[i][1] - segs[i][0]
            idx = i

    return [segs[idx][0], segs[idx][1]]


def vertical_segmentation(image):
    segs = segmentation(image, 0)

    chiffres = []

    for i in range(len(segs)):
        if (segs[i][1] - segs[i][0]) > 1:
            chiffres.append([segs[i][0], segs[i][1]])

    return chiffres