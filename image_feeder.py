import cv2
import numpy as np
import requests
import json

image = cv2.imread('image.jpg')
image_np_expanded = np.expand_dims(image, axis=0)



res = requests.post('http://192.168.1.2:8051/predict', json={ "instances": json.dumps(image_np_expanded.tolist())})
if res.ok:
    print(res.json())
