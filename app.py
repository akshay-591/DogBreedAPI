_author_ = 'Akshay Kumar'
_Git_ = 'https://github.com/akshay-591'


from flask import Flask, jsonify,request

import base64
from PIL import Image
from io import BytesIO
import numpy as np

from tensorflow.keras.applications import resnet50
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.resnet import preprocess_input


app = Flask(__name__)

@app.route("/",methods = ["POST"])
def ProcessData():
    data = request.get_json(force=True) # take base64 
    image_filename = data['image']

    im_bytes = base64.b64decode(image_filename.encode())   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    img.thumbnail(size = (224,224))
    # convert to array
    img_array = np.array([img_to_array(img)])
    # scale and return
    output = preprocess_input(img_array)

    my_model =resnet50.ResNet50(weights = 'imagenet')
    preds = my_model.predict(output)
    most_likely_labels = resnet50.decode_predictions(preds,top=1)
    labels = jsonify({'breed' : most_likely_labels[0][0][1],
                      'score' : str(most_likely_labels[0][0][2])})
    print(labels)
    return labels

if __name__ == '_main_':
    app.run(debug=True)