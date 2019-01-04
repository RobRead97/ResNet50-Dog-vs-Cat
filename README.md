# ResNet50 Transfer Learning Dog vs Cat
A convolutional neural network that can classify picture of dogs and cats.
The Model has two layers. The first layer is the pre-trained ResNet-50 model, which is very powerful
and can classify objects into one of thousands of categories. The second layer is a Dense layer with a softmax
activation function.

# Dataset
The training dataset used contains 10,000 images of cats and 10,000 images of dogs.

The testing dataset used contains 2,500 images of cats and 2,500 images of dogs.

Also Contains an unlabelled dataset of 12,500 dog and cat pictures to use for predictions.

# Model
On my initial run using SGD as my optimization function, and a batch size of 25, the results were the following:
```
2838s 4s/step - loss: 0.1019 - acc: 0.9620 - val_loss: 0.0607 - val_acc: 0.9776
```

That's 47 minutes and 18 seconds to train, with an accuracy of 97.76%

Of course I ran the training on a potato and therefor training time would be shorter with a better computer. Still,
this dataset is obviously more than sufficient. In future iterations I will try to apply Data Augmentation as well as 
Strides and Dropout to see if I can reach higher than 99% accuracy.

# Latest Run

```
1278s 2s/step - loss: 0.0991 - acc: 0.9626 - val_loss: 0.0543 - val_acc: 0.9802
```


