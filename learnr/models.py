import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.optimizers as ko

def Ribli19(imsize=51, n_target=2 ,n_channels=1, nf=64, reg = 5e-5,
          padding='same'):

  """
  Creates a keras model based on Ribli et al 2019 (originally used to train on ellipticities)
  
  Args:
    imsize: int
      size in pixels of stamp image
    n_target: int
      number of outputs
    n_channels: int
      number of input images (eg. different epochs, filters)
    nf: int
      number of convolution filters in each layer (2^n)
    reg: float
      convolution kernel regularization parameter
    padding: string
      padding for convolution kernel
  Returns:
    model: keras model
  
  """
  #input
  inp = kl.Input((imsize, imsize,n_channels))

  # conv block 1
  x = kl.Conv2D(nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(inp)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 2
  x = kl.Conv2D(2*nf, (3, 3), padding=padding,kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 3
  x = kl.Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(2*nf, (1, 1), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))
  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 4
  x = kl.Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(4*nf, (1, 1), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))
  x = kl.MaxPooling2D(strides=(2,2))(x)

  # conv block 5
  x = kl.Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(8*nf, (1, 1), padding=padding,  kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  x = kl.Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=kr.l2(reg))(x)
  x = kl.Activation('relu')(kl.BatchNormalization()(x))

  #  end of conv

  x = kl.GlobalAveragePooling2D()(x)    
  x = kl.Dense(n_target)(x)#, name = 'final_dense_n%d_ngpu%d' % (n_target, len(gpu.split(','))))(x)  

  model = km.Model(inputs=inp, outputs=x)  # make model

  return model