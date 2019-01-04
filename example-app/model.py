from tensorflow.python.keras.models import model_from_json
import cv2
import numpy as np


# This class encapsulates a pre-trained model and is responsible for the loading of that model from disk.
class MyModel:
    def __init__(self, path_to_model, path_to_weights, loss, optimizer, img_width=224, img_height=224):
        self.img_width = img_width
        self.img_height = img_height

        # Load the model from json file
        json_file = open(path_to_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights(path_to_weights)
        self.loaded_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Loaded model from disk")

    def predict(self, img_path):
        # Process the image into a numpy 4D array
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.reshape(img, [1, self.img_width, self.img_height, 3])

        # Send the numpy array as a tensor to the model and convert the class data to human readable strings.
        prediction = self.loaded_model.predict_classes(img)
        if prediction[0] == 0:
            return 'Cat'
        else:
            return 'Dog'
