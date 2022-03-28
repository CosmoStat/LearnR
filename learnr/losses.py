import tensorflow as tf
from autometacal import get_metacal_response_finitediff

@tf.function
def loss_fd(batch,model,reconv_psf_image,step=0.01):
  """
   Defines a loss respective to unit shear response
   
  Args:
    batch: tf batch 
      Image stamps as ['obs_image'] and psf models as ['psf_image']
    reconv_psf_image: tf tensor 
      Synthetic reconvolution psf
    step: float
      Step size for the finite differences
     
  Returns:
    lost: float
      Distance between the shear response matrix and unity.
    
  """
  
  shears = tf.random.uniform((batch_size,2),-.1,.1,dtype=tf.float32)
  #compute response
  R = get_metacal_response_finitediff(batch['obs_image'],
                                      batch['psf_image'],
                                      reconv_psf_image,
                                      shear=shears,
                                      step=step,
                                      method=model)[1]

  
  lost = tf.norm(R - tf.eye(2))
  
  return lost



@tf.function
def loss_ad(batch,model,reconv_psf_image,step=0.01):
  """
   Defines a loss respective to unit shear response
   
  Args:
    batch: tf batch 
      Image stamps as ['obs_image'] and psf models as ['psf_image']
    reconv_psf_image: tf tensor 
      Synthetic reconvolution psf
     
  Returns:
    lost: float
      Distance between the shear response matrix and unity.
    
  """
  print('WARNING! This does not work!')
  shears = tf.random.uniform((batch_size,2),-.1,.1,dtype=tf.float32)
  #compute response
  R = get_metacal_response(batch['obs_image'],
                                      batch['psf_image'],
                                      reconv_psf_image,
                                      shear=shears,
                                      method=model)[1]

  
  lost = tf.norm(R - tf.eye(2))
  
  return lost


