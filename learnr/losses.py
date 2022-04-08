import tensorflow as tf
from autometacal import get_metacal_response_finitediff

@tf.function
def loss_fd(obs_batch,psf_batch,reconv_psf,model,shear_range=.1):
  """
   Defines a loss respective to unit shear response
   
  Args:
    obs, psf _batch: tf batch 
      obs images and psfs
      
    reconv_psf_image: tf tensor 
      Synthetic reconvolution psf
    step: float
      Step size for the finite differences
     
  Returns:
    lost: float
      Distance between the shear response matrix and unity.
  """
  
  shears = tf.random.uniform((batch_size,2),-shear_range,shear_range,dtype=tf.float32)
  #compute response
  R = get_metacal_response_finitediff(obs_batch,
                                      psf_batch,
                                      reconv_psf,
                                      shear=shears,
                                      step=0.01,
                                      method=model)[1]
  #R = amc.get_metacal_response(...)[1]
  
  lost = tf.norm(R - tf.eye(2),ord=2)
  
  return lost


@tf.function
def loss_fd_real(obs_batch,psf_batch,reconv_psf,model,shear_range=.1):
  """
   Defines a loss respective to unit shear response
   
  Args:
    obs, psf _batch: tf batch 
      obs images and psfs
      
    reconv_psf_image: tf tensor 
      Synthetic reconvolution psf
    step: float
      Step size for the finite differences
     
  Returns:
    lost: float
      Distance between the shear response matrix and unity.
  """
  
  shears = tf.random.uniform((batch_size,2),-shear_range,shear_range,dtype=tf.float32)
  #compute response
  R = get_metacal_response_finitediff(obs_batch,
                                      psf_batch,
                                      reconv_psf,
                                      shear=shears,
                                      step=0.01,
                                      method=model)[1]
  #R = amc.get_metacal_response(...)[1]
  
  lost = tf.norm(R - tf.eye(2),ord=2)
  
  return lost


