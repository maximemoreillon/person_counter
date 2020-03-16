import os
import tensorflow as tf
from flask import Flask, request
import numpy as np

# Model download
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
LOCAL_FILE_NAME = '{}.tar.gz'.format(MODEL_NAME)
MODELS_DIRECTORY = '{}/models/'.format(os.getcwd())
MODEL_URL = 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(MODEL_NAME)
MODEL_PATH = '{}/models/{}/saved_model'.format(os.getcwd(), MODEL_NAME)

print('Downloading model ...')
model_dir = tf.keras.utils.get_file(
    fname=LOCAL_FILE_NAME,
    origin=MODEL_URL,
    cache_subdir=MODELS_DIRECTORY,
    archive_format='tar',
    extract=True)
print('Downloading model OK')

# Model loading
print('Loading model ...')
model = tf.saved_model.load(MODEL_PATH)
model = model.signatures['serving_default']
print('Loading model OK')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Object detection microservice'


@app.route('/predict', methods=['GET','POST']) # Change to only POST when done
def predict():

    print('Starting count')

    content = request.json
    # This is super slow
    image = eval('np.array({})'.format(content['instances']))

    print(image.shape)

    # conversion to tensor
    image_tensor = tf.convert_to_tensor(image, dtype=np.uint8)

    # Feed the image to the network
    output_dict = model(image_tensor)

    # Process the output
    classes_numpy = output_dict['detection_classes'].numpy()
    scores_numpy = output_dict['detection_scores'].numpy()
    classes = [int(x) for x in classes_numpy[0].tolist()]
    scores = scores_numpy[0].tolist()

    DETECTION_THRESHOLD = 0.5

    person_count = 0

    for i, object_class in enumerate(classes):
        if  object_class == 1 and scores[i] > DETECTION_THRESHOLD:

            # Count is done here
            person_count = person_count + 1

    print("Person count: {}".format(person_count))

    return {"count": person_count}

print('Starting server')
app.run(host='0.0.0.0', port=8051)
