"""
implementation of basic UNet architecture, tested on Covid-19 CT image dataset, axial slices of lungs
https://www.kaggle.com/competitions/covid-segmentation/overview
https://arxiv.org/pdf/1505.04597.pdf
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
"""
class UNetSegmentation(tf.keras.Model):
    def __init__(self,):
        super()
        
        self.Conv3 = 
        
    def call():
        return 
"""

def Conv3(inp, channels):
    out = tf.keras.layers.Conv2D(filters = channels, 
                                 kernel_size = (3,3), 
                                 padding = 'same', 
                                 activation = "relu")(inp)
    return out
    
def UNetDownStep(inp, channels):
    levela = Conv3(inp, channels)
    levelb = Conv3(levela, channels)
    
    out = tf.keras.layers.MaxPool2D((2,2))(levelb)
    
    return levelb, out

def UNetDownChannel(inp):
    intermediate1, step1 = UNetDownStep(inp, 64)
    intermediate2, step2 = UNetDownStep(step1, 128)
    intermediate3, step3 = UNetDownStep(step2, 256)
    intermediate4, step4 = UNetDownStep(step3, 512)
    
    step5 = Conv3(step4, 1024)
    step5 = Conv3(step5, 1024)
    
    return intermediate1, intermediate2, intermediate3, intermediate4, step5

def UpConv(inp, channels):
    
    #method 1, upsample + conv2d
    ups = tf.keras.layers.UpSampling2D((2,2), )(inp)

    return tf.keras.layers.Conv2D(channels, 
                                  kernel_size = (2,2), 
                                  padding = 'same', 
                                  activation = 'relu')(ups)
    #method 1, transpose convolution

def UNetUpStep(inp, intermediate, channels):
    prev_layer = UpConv(inp, channels[0])
    #crop intermediate
    cropped = tf.keras.layers.Cropping2D((intermediate.shape[1]//2 - prev_layer.shape[1]//2,
                                intermediate.shape[2]//2 - prev_layer.shape[2]//2))(intermediate)
    l1 = tf.keras.layers.concatenate([cropped, prev_layer], axis = 3)
    l2 = Conv3(l1, channels[1] )
    l3 = Conv3(l2, channels[1])
    
    return l3

def UNetUpChannel(l1, l2, l3, l4, l5):
    
    #want to crop l4 somehow
    step1 = UNetUpStep(l5, l4, [1024,512])
    step2 = UNetUpStep(step1, l3, [512,256])
    step3 = UNetUpStep(step2, l2, [256,128])
    step4 = UNetUpStep(step3, l1, [128,64])
    
    return step4 
    
def UNet(inp_shape, classes):
    inp = tf.keras.Input(inp_shape)
    
    l1, l2, l3, l4, l5 = UNetDownChannel(inp)
    preseg = UNetUpChannel(l1, l2, l3, l4, l5)
    segmented_layer = tf.keras.layers.Conv2D(filters = classes, 
                                             kernel_size = (1,1), 
                                             padding = 'same',
                                             activation = 'sigmoid')(preseg)
    
    UNetModel = tf.keras.Model(inputs = inp, outputs = segmented_layer, )
    #model.compile(optimizer = adam, loss = binary_crossentropy, metrics = ['accuracy'])
    return UNetModel

def VisualizeResults(model, test_data):
    sample1 = int(random.random()*len(test_data))
    sample2 = int(random.random()*len(test_data))
    
    if sample1==sample2:
        sample2+=1
    
    pred = model(np.concatenate([np.expand_dims(test_data[sample1], 0), 
                    np.expand_dims(test_data[sample2], 0)]))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(np.argmax(pred[0], axis = 2))
    plt.title('image 1 pred')
    plt.subplot(2,2,2)
    plt.imshow(test_data[sample1])
    plt.title('image 1 input')
    
    plt.subplot(2,2,3)
    plt.imshow(np.argmax(pred[1], axis = 2))
    plt.title('image2 pred')
    plt.subplot(2,2,4)
    plt.imshow(test_data[sample2])
    plt.title('image 2 input')
    
    plt.show()
    
    return

def DisplayTrainingTrend(model):
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(model.history.history['loss'])
    plt.title('Loss over training')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    
    plt.subplot(1,2,2)
    plt.plot(model.history.history['accuracy'])
    plt.title('accuracy over training')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy (%)')
    
    plt.show()
    
    return
    


m = UNet((512,512,1),4)
m.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), 
          loss = 'binary_crossentropy', 
          metrics = ['accuracy'])

train = np.load('E:/CovidCTImageSeg/covid-segmentation/images_medseg.npy')
target = np.load('E:/CovidCTImageSeg/covid-segmentation/masks_medseg.npy')

batch_size = 2
epochs = 50

m.fit(train, target, batch_size = batch_size, epochs = epochs, shuffle = True)

test = np.load('E:/CovidCTImageSeg/covid-segmentation/test_images_medseg.npy')



#saving weights
#tf.keras.callbacks.ModelCheckpoint -> callback allows continually saving weight during and @ end of training
#   https://www.tensorflow.org/tutorials/keras/save_and_load

#cehckpoint_path = "..../cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1)
#model.fit(...., callbacks = [cp_callback])

