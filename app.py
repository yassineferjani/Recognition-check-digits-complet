import numpy as np
from flask import Flask
from flask import render_template
import os
from flask import request


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from prediction import *
from preprocess import *

app = Flask(__name__)
UPLOAD_FOLDER = "C:\\Users\\yassi\PycharmProjects\\flaskProject\\static\\images"

def resultPreparation (montant, precision):
    chiffres=montant
    taux=precision
    virgule=[]
    while (chiffres[-1] == '.' or chiffres[0] == '.' or chiffres[0] == '0'):
        if chiffres[-1] == '.':
            chiffres.pop()
            taux.pop()
        if chiffres[0] == '.':
            chiffres.pop(0)
            taux.pop(0)

    for i in range(len(chiffres)):
        if chiffres[i] == '.':
            virgule.append(i)
    chaine = ''.join(map(str,np.delete(chiffres, virgule[: -1])))
    while chaine[0] == '0':
        chaine = chaine[1:]
        taux.pop(0)

    taux=np.delete(taux,virgule[: -1])
    taux_precision= np.mean(taux)
    return chaine, taux_precision


@app.route('/', methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

        ''' 
        in_memory_file = io.BytesIO()
         image_file.save(in_memory_file)
         data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
         color_image_flag = 1
         img = cv2.imdecode(data, color_image_flag)
         '''
        img = cv2.imread(image_location,0)
        montant = money_region(img)
        img_prepared = preprocess(montant)
        segment,_= horizontal_segment(img_prepared)
        orig = horizontal_segmentimage1(montant, img_prepared)

        montant, taux = prediction(segment,orig)

        chiffre, pres=resultPreparation(montant,taux)
        pres= round(pres*100,2)

        return render_template("index.html", prediction=chiffre, precision=pres, imagelocation=image_file.filename)


    return render_template("index.html", prediction=None, precision=None, imagelocation=None)

if __name__ == '__main__':
    app.run()
