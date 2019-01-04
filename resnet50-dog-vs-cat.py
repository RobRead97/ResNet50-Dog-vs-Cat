from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

num_classes = 2  # Dogs or Cats obviously
resnet_weights_path = './ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_path = "./model/"

# The Model I'm building has two layers. The first layer is the pre-trained ResNet-50 model, which is very powerful
# and can classify objects into one of thousands of categories. The second layer is a Dense layer with a softmax
# activation function. This layer contains two neurons, one for Dog and one for Cat, and the weights trained using
# our dataset.
seq_model = Sequential()
seq_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
seq_model.add(Dense(num_classes, activation='softmax'))

# Since the ResNet model is already trained, I won't train it.
seq_model.layers[0].trainable = False

# I'm using fairly small batch sizes, so I will use Stochastic Gradient Descent. Also I want to see the accuracy
# metric while I'm training. Love seeing that number slowly climb up while the model is training on my potato.
seq_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Reads the images from our training set. Batch size of 25 gives us 800 batches. I've already categorized the set.
train_generator = data_generator.flow_from_directory(
        './train',
        target_size=(image_size, image_size),
        batch_size=25,
        class_mode='categorical')

# Reads the images from our training set. Also already categorized.
validation_generator = data_generator.flow_from_directory(
        './test',
        target_size=(image_size, image_size),
        class_mode='categorical')

# Train the model. On my potato, this takes about 45 minutes. Currently the model is 98.02% accurate.
seq_model.fit_generator(
        train_generator,
        validation_data=validation_generator)

# Save the model to disk for future use.
model_json = seq_model.to_json()
with open(model_path + "dogs_vs_cats_classifier.json", "w") as json_file:
    json_file.write(model_json)

seq_model.save_weights(model_path + "dogs_vs_cats_classifier.h5")
print("Saved model to disk")
