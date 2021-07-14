import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from prediction import *
from preprocess import *
import matplotlib.pyplot as plt


img = cv2.imread("C:/Users/yassi/Desktop/BH/BH1.jpg",0)
montant = money_region(img)

img_prepared = preprocess(montant)
img1,_= horizontal_segment(img_prepared)

orig=horizontal_segmentimage1(montant,img_prepared)
plt.imshow(orig)
plt.show()
old_height= img1.shape[0]


pix_density = []
(h, w) = img_prepared.shape

for idx in range(w):
    col = img_prepared[0: h, idx: idx + 1]
    pix_density.append(np.sum(col))

pix_density = np.sum(img_prepared, axis=1, keepdims=True)

x = list(np.nonzero(pix_density)[0])
print(x)
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






"""plt.figure(figsize=(50, 50))
res, pos = segmentChiffre(img_prepared, orig)

mean_upper = np.mean([pt[0] for pt in pos])
mean_lower = np.mean([pt[1] for pt in pos])
mean_height = np.mean([pt[0] - pt[1] for pt in pos])

for i in range(len(res)):
    try:

        plt.subplot(((len(res) // 2) + 1), ((len(res) // 2) + 1), i + 1)
        plt.xticks([])
        plt.yticks([])

        dil = res[i]

        # dil = cv2.dilate(dil, kernel)

        h1, w1 = dil.shape

        if (h1 + 10 >= w1):

            r = h1 / w1
            dil = cv2.resize(dil, (int(np.ceil(30 / r)), 30))
        else:

            r = w1 / h1
            dil = cv2.resize(dil, (30, int(np.ceil(30 / r))))

        h, w = dil.shape
        yoff = round((64 - h) / 2)
        xoff = round((64 - w) / 2)
        stk[yoff:yoff + h, xoff:xoff + w] = dil

        stk = stk / 255.

        pred_img = stk.reshape(1, 64, 64, 1)
        title = np.argmax(model_one.predict(pred_img))

        if (pos[i][1] < old_height / 2):
            title = ''

        if (title == 1) and (abs(pos[i][0] - mean_lower) < abs(pos[i][0] - mean_upper) or (h1 < mean_height)):
            title = ','

        plt.title(f'{title} Taux: {np.max(model_one.predict(pred_img))}')
        plt.imshow(stk, 'gray')

    except:
        continue"""

"""segment = []
position = []
kernel = np.ones((3, 3), dtype=np.int8)
dil = cv2.dilate(img_prepared, kernel)
res = vertical_segmentation(dil)
for i in range(len(res)):
    segment.append(horizontal_segment(orig[:, res[i][0]:res[i][1]])[0])
    position.append(horizontal_segment(orig[:, res[i][0]:res[i][1]])[1])"""



