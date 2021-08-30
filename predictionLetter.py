from keras.models import load_model
import string
import difflib
import cv2
import numpy as np




list_lettres = list(string.ascii_lowercase)
model= load_model("/content/gdrive/MyDrive/PKL TRAIN/Letters/Model_0508.h5")

dict_mots = ['un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize', 'vingt',
'trente', 'quarante', 'cinquante', 'soixante', 'cent', 'mille', 'million',
'milliard', 'billion', 'dinars', 'millimes', 'et']

def predictionMot(imagemot,listletters):
  plt.figure(figsize=(50,50))
  mot=[]
  kernel= np.ones((3,3))
  for i in range(len(listletters)-1):
    stk = np.zeros((64, 64))
    plt.subplot(((len(listletters)-1)), 1, i+1)
    plt.xticks([])
    plt.yticks([])

    indiceim=horizontal_segment(imagemot[:,listletters[i]:listletters[i+1]])
    img=imagemot[indiceim[0]:indiceim[-1],listletters[i]:listletters[i+1]]
    #img= cv2.ximgproc.thinning(img)
    #img = cv2.dilate(img, kernel)
    #img=cv2.medianBlur(img,3)

    h, w = img.shape

    if h > w :
      r= h/w
      dil= cv2.resize(img,(round(50/r),50))
    elif h==w:
      dil= cv2.resize(img,(50, 50))

    else:
      r= w/h
      dil= cv2.resize(img,(50,round(50/r)))
    h1, w1 = dil.shape
    yoff = round((64-h1)/2)
    xoff = round((64-w1)/2)
    stk[yoff:yoff+h1, xoff:xoff+w1] = dil

    stk = stk / 255.


    pred_img = stk.reshape(1, 64, 64, 1)

    plt.title(f'{np.argmax(model.predict(pred_img))}' )
    plt.imshow(stk, 'gray')
    mot.append(np.argmax(model.predict(pred_img)))

  return mot

