from model import MyModel
from PIL import Image, ImageFont, ImageDraw
import random
import sys

# Can specify the number of random images to predict. default is 4
if len(sys.argv) > 1:
    num_to_predict = int(sys.argv[1])
else:
    num_to_predict = 4

model_path = '../model/dogs_vs_cats_classifier.json'
weights_path = '../model/dogs_vs_cats_classifier.h5'
loss = 'categorical_crossentropy'
optimizer = 'sgd'
img_paths = ['../predict/' + str(random.randint(1, 12500)) + '.jpg' for i in range(num_to_predict)]

# The pre-trained model, loaded from disk.
model = MyModel(model_path, weights_path, loss, optimizer)


# This will first get a prediction from the trained model, then display the photo with labelled with the prediction
def show_img_with_prediction(img_path):
    prediction = model.predict(img_path)
    with Image.open(img_path) as img:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", 36)
        draw.text((12, 12), prediction, (255, 255, 255), font=font)
        img.show()


# Show the images with predictions
for i, path in enumerate(img_paths):
    show_img_with_prediction(path)
