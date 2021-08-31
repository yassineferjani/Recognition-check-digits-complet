from preprocess import *
from keras.models import load_model

model_length = load_model('LengthModel0908.h5')
model_one= load_model('modelonedigitupdated.h5')
model_two= load_model('model_two_1407_updated.h5')
model_Three= load_model('chiffre3Model1307_complet.h5')


def segmentChiffre(image, orig):
    segment = []
    position = []
    kernel = np.ones((3, 3), dtype=np.int8)
    dil = cv2.dilate(image, kernel)
    res = vertical_segmentation(dil)

    for i in range(len(res)):
        chiffre = horizontal_segment(orig[:, res[i][0]:res[i][1]])[0]
        h, w = chiffre.shape
        if h * w > 10:
            segment.append(chiffre)
            position.append(horizontal_segment(orig[:, res[i][0]:res[i][1]])[1])

    return segment, position

def LengthPrediction(digit, model=model_length):

    dil = cv2.copyMakeBorder(digit, 7, 7, 7, 7, cv2.BORDER_CONSTANT)
    dil = cv2.resize(dil, (64, 64))

    pred_img = dil.reshape(1, 64, 64, 1)
    pred = model.predict(pred_img, batch_size=1)

    return pred


def Pred_one_digit(digit, model=model_one):
    try:

        stk = np.zeros((64, 64))

        h, w = digit.shape

        if (h + 10 >= w):

            r = h / w
            dil = cv2.resize(digit, (int(np.ceil(35 / r)), 35))
        else:

            r = w / h
            dil = cv2.resize(digit, (35, int(np.ceil(35 / r))))

        h, w = dil.shape
        yoff = round((64 - h) / 2)
        xoff = round((64 - w) / 2)
        stk[yoff:yoff + h, xoff:xoff + w] = dil

        stk = stk / 255.

        pred_img = stk.reshape(1, 64, 64, 1)
        return np.argmax(model.predict(pred_img)), np.max(model.predict(pred_img))
    except:
        return None, None



def Pred_two_digit(digit, model=model_two):
    h1, w1 = digit.shape
    stk = np.zeros((64, 64))

    if h1 >= w1:

        r = h1 / w1
        dil = cv2.resize(digit, (int(np.ceil(55 / r)), 55))
    else:

        r = w1 / h1
        dil = cv2.resize(digit, (55, int(np.ceil(55 / r))))

    h, w = dil.shape

    yoff = round((64 - h) / 2)
    xoff = round((64 - w) / 2)

    stk[yoff:yoff + h, xoff:xoff + w] = dil
    stk = stk / 255.
    pred_img = stk.reshape(1, 64, 64, 1)

    return np.argmax(model.predict(pred_img)), np.max(model.predict(pred_img))

def Pred_three_digit(digit, model=model_Three):
    stk = np.zeros((64, 64))
    h1, w1 = digit.shape
    if h1 > w1:
        r = h1 / w1
        dil = cv2.resize(digit, (int(np.ceil(55 / r)), 55))

    else:
        r = w1 / h1
        dil = cv2.resize(digit, (55, int(np.ceil(55 / r))))

    h, w = dil.shape
    yoff = round((64 - h) / 2)
    xoff = round((64 - w) / 2)
    stk[yoff:yoff + h, xoff:xoff + w] = dil

    stk = stk / 255.

    pred_img = stk.reshape(1, 64, 64, 1)

    return np.argmax(model.predict(pred_img)), np.max(model.predict(pred_img))


def prediction(image, orig):
    old_height=orig.shape[0]
    digits, pos = segmentChiffre(image, orig)
    mean_upper = np.mean([pt[0] for pt in pos])
    mean_lower = np.mean([pt[1] for pt in pos])
    mean_height = np.mean([pt[1] - pt[0] for pt in pos])

    predictions = []
    taux = []
    for idx in range(len(pos)):
        if pos[idx][1] - pos[idx][0] >= mean_height:
            x = idx
            break
    for i in range(x,len(digits)):

        len_dig = LengthPrediction(digits[i])

        if np.argmax(len_dig) == 0:
            pred, tau = Pred_one_digit(digits[i])
            if (pos[i][1] < old_height / 2):
                pred = ''
            if (abs(pos[i][0] - mean_lower) < abs(pos[i][0] - mean_upper)):
                pred = '.'
            predictions.append(pred)
            taux.append(tau)

        elif np.argmax(len_dig) == 1:
            pred, tau = Pred_two_digit(digits[i])
            if pred in list(range(10)):
                pred='0'+ str(pred)
            predictions.append(pred)
            taux.append(tau)

        elif np.argmax(len_dig) == 2:
            pred, tau = Pred_three_digit(digits[i])
            if pred in list(range(10)):
                pred='00'+ str(pred)
            if pred in list(range(10,100)):
                pred = '0' + str(pred)
            predictions.append(pred)
            taux.append(tau)


    return predictions, taux