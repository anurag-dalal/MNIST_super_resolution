# MNIST_super_resolution
This repository contain up-sampling 4x using GAN for super resolution in the MNIST dateset.

## Install Dependecies
```unix
$pip install numpy
$pip install matplotlib
$pip install tensorflow
$pip install scikit-learn
$pip install keras
$pip install opencv-python
```
## Imporant reads
[This](https://arxiv.org/abs/1609.04802) paper is the main inspiration in super resolution using GAN. In this project a small generating network is used with a very small feature based discriminator.

[This](https://www.geeksforgeeks.org/super-resolution-gan-srgan/) is a very simple and straightforward explanation of how GAN are used in supresolution. Note that MNIST is a very simple dataset and simple MSE loss is used for both Generator and Discriminator.

[This](https://medium.com/@ramyahrgowda/srgan-paper-explained-3d2d575d09ff) is a very nice explanation of the SRGAN paper.

## The generator model used is:
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            [(None, 7, 7, 1)]    0                                            
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 7, 7, 32)     320         input_3[0][0]                    
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 7, 7, 32)     128         conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 7, 7, 32)     0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 7, 7, 32)     9248        activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 7, 7, 32)     128         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 7, 7, 32)     0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
add_4 (Add)                     (None, 7, 7, 32)     0           activation_12[0][0]              
                                                                 activation_13[0][0]              
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 7, 7, 32)     9248        add_4[0][0]                      
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 7, 7, 32)     128         conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 7, 7, 32)     0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
add_5 (Add)                     (None, 7, 7, 32)     0           add_4[0][0]                      
                                                                 activation_14[0][0]              
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 7, 7, 32)     0           add_5[0][0]                      
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 14, 14, 32)   0           activation_15[0][0]              
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 14, 14, 32)   9248        up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 14, 14, 32)   128         conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 14, 14, 32)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 14, 14, 32)   9248        activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 14, 14, 32)   128         conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 14, 14, 32)   0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
add_6 (Add)                     (None, 14, 14, 32)   0           activation_16[0][0]              
                                                                 activation_17[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 14, 14, 32)   9248        add_6[0][0]                      
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 14, 14, 32)   128         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 14, 14, 32)   0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
add_7 (Add)                     (None, 14, 14, 32)   0           add_6[0][0]                      
                                                                 activation_18[0][0]              
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 14, 14, 32)   0           add_7[0][0]                      
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 28, 28, 32)   0           activation_19[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 28, 28, 32)   9248        up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 28, 28, 32)   128         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 28, 28, 32)   0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 28, 28, 1)    289         activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 28, 28, 1)    4           conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 28, 28, 1)    0           batch_normalization_16[0][0]     
==================================================================================================
Total params: 56,997
Trainable params: 56,547
Non-trainable params: 450
__________________________________________________________________________________________________
```

## The discriminator model used is:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 28, 28, 32)        320       
_________________________________________________________________
activation_22 (Activation)   (None, 28, 28, 32)        0         
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 28, 28, 64)        18496     
_________________________________________________________________
batch_normalization_17 (Batc (None, 28, 28, 64)        256       
_________________________________________________________________
activation_23 (Activation)   (None, 28, 28, 64)        0         
=================================================================
Total params: 19,072
Trainable params: 18,944
Non-trainable params: 128
_________________________________________________________________
```

## Loading the data
The data is loaded from keras dataset. Only the images are used, and the labels are discarded.
```python
(Y_train, _), (Y_test, _) = mnist.load_data()
```

## Preprocessing the data
The loaded data are actually the high resolution image of 28x28.  
The images are first downsample by using a function, and a gaussina noise with μ = 0, and σ = 10 is introduced. These become our input data.  
Then there is a addition of dimension in the data as tensorflow convolution filters works on 3-dim tensors.
```python
# downsampling and introducing gaussian noise
# this downsampled and noised dataset is out X or inputs
X_train = downSampleAndNoisyfi(Y_train)
X_test = downSampleAndNoisyfi(Y_test)

# introduce a new dimension to the data (None, 28, 28, 1)
X_test = X_test[..., np.newaxis]
X_train = X_train[..., np.newaxis]
Y_train = Y_train[..., np.newaxis]
Y_test = Y_test[..., np.newaxis]

```

## Training the model
The model is trained for a batch size of 100 for 50 epochs. 

## Results
LR = Low Resolution.  
SR = Predicted Super Resolution.  
HR = Original High Resolution.  

![First five image from test](/images/SR.png "First five image from test")
![Random five image from test](/images/SR_random.png "Random five image from test")
