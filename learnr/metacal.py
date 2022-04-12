import galflow as gf
import tensorflow as tf
from autometacal import generate_mcal_image

def get_metacal_response_finitediff(gal_image,
                                    psf_image,
                                    reconv_psf_image,
                                    shear,
                                    step,
                                    method):
  """
  Gets shear response as a finite difference operation, 
  instead of automatic differentiation.
  """
  gal_image = tf.convert_to_tensor(gal_image,dtype=tf.float32)
  psf_image = tf.convert_to_tensor(gal_image,dtype=tf.float32)
  batch_size, _ , _ = gal_image.get_shape().as_list()
  step_batch = tf.constant(step,shape=(batch_size,1),dtype=tf.float32)
  
  step1p = tf.pad(step_batch,[[0,0],[0,1]])
  step1m = tf.pad(-step_batch,[[0,0],[0,1]])
  step2p = tf.pad(step_batch,[[0,0],[1,0]])
  step2m = tf.pad(-step_batch,[[0,0],[1,0]])
    
  img0s = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    tf.zeros([batch_size,2])
  ) 
  
  shears1p = shear + step1p
  img1p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears1p
  )
  
  shears1m = shear + step1m 
  img1m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears1m
  ) 
  
  shears2p = shear + step2p 
  img2p = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears2p
  )
  
  shears2m = shear + step2m 
  img2m = generate_mcal_image(
    gal_image,
    psf_image,
    reconv_psf_image,
    shears2m
  ) 
  
  g0s = method(img0s)
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  R11 = (g1p[:,0]-g1m[:,0])/(2*step)
  R21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  R12 = (g2p[:,0]-g2m[:,0])/(2*step)
  R22 = (g2p[:,1]-g2m[:,1])/(2*step)
 
  #N. B.:The matrix is correct. 
  #The transposition will swap R12 with R21 across a batch correctly.
  R = tf.transpose(tf.convert_to_tensor(
    [[R11,R21],
     [R12,R22]],dtype=tf.float32)
  )
  
  ellip_dict = {
    'noshear':g0s,
    '1p':g1p,
    '1m':g1m,
    '2p':g2p,
    '2m':g2m,    
  } 

  return ellip_dict, R

def generate_real_mcal_image(
  gal_images,
  reconvolution_psf_image,
  g
):
  """ Generate a metacalibrated image given input and target PSFs.
  
  Args: 
    gal_images: tf.Tensor or np.array
      (batch_size, N, N ) image of galaxies
    psf_images: tf.Tensor or np.array
      (batch_size, N, N ) image of psf model
    reconvolution_psf_image: tf.Tensor
      (N, N ) tensor of reconvolution psf model
    g: tf.Tensor or np.array
    [batch_size, 2] input shear
  Returns:
    img: tf.Tensor
      tf tensor containing image of galaxy after deconvolution by psf_deconv, 
      shearing by g, and reconvolution with reconvolution_psf_image.
  
  """
  #cast stuff as float32 tensors
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)  
  #psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32) 
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)  
  g = tf.convert_to_tensor(g, dtype=tf.float32)  
  
  #Get batch info
  batch_size, nx, ny = gal_images.get_shape().as_list()  
      
  #add pads in real space
  padfactor = 3 #total width of image after padding
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
    
  padded_gal_images = tf.pad(gal_images,paddings)
  padded_psf_images = tf.pad(psf_images,paddings)
  padded_reconvolution_psf_image = tf.pad(reconvolution_psf_image,paddings)
    
  #Convert galaxy images to k space
  im_shift = tf.signal.ifftshift(padded_gal_images,axes=[1,2]) # The ifftshift is to remove the phase for centered objects
  im_complex = tf.cast(im_shift, tf.complex64)
  im_fft = tf.signal.fft2d(im_complex)
  imk = tf.signal.fftshift(im_fft, axes=[1,2])#the fftshift is to put the 0 frequency at the center of the k image
  
  #Convert psf images to k space  
  #psf_complex = tf.cast(padded_psf_images, tf.complex64)
  #psf_fft = tf.signal.fft2d(psf_complex)
  #psf_fft_abs = tf.abs(psf_fft)
  #psf_fft_abs_complex = tf.cast(psf_fft_abs,tf.complex64)
  #kpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])

  #Convert reconvolution psf image to k space 
  rpsf_complex = tf.cast(padded_reconvolution_psf_image, tf.complex64)
  rpsf_fft =  tf.signal.fft2d(rpsf_complex)
  rpsf_fft_abs = tf.abs(rpsf_fft)
  psf_fft_abs_complex = tf.cast(rpsf_fft_abs,tf.complex64)
  krpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype='complex64')
  mask = tf.expand_dims(mask, axis=0)

  # Deconvolve image from input PSF
  im_deconv = imk * mask #* ( (1./(kpsf+1e-10))*mask)

  # Apply shear
  im_sheared = gf.shear(tf.expand_dims(im_deconv,-1), g[...,0], g[...,1])[...,0]

  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(im_sheared * krpsf * mask))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))
  
  # Add noise
  img = img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]
  img += tf.random.normal([nx,ny],0,1e-6)

  return img

def get_real_metacal_response_finitediff(gal_image,
                                    #psf_image,
                                    reconv_psf_image,
                                    shear,
                                    step,
                                    method):
  """
  Gets shear response as a finite difference operation, 
  instead of automatic differentiation.
  """
  batch_size, _ , _ = gal_image.get_shape().as_list()
  step_batch = tf.constant(step,shape=(batch_size,1),dtype=tf.float32)
  
  step1p = tf.pad(step_batch,[[0,0],[0,1]])
  step1m = tf.pad(-step_batch,[[0,0],[0,1]])
  step2p = tf.pad(step_batch,[[0,0],[1,0]])
  step2m = tf.pad(-step_batch,[[0,0],[1,0]])
    
  img0s = generate_real_mcal_image(
    gal_image,
    #psf_image,
    reconv_psf_image,
    tf.zeros([batch_size,2])
  ) 
  
  shears1p = shear + step1p
  img1p = generate_real_mcal_image(
    gal_image,
    #psf_image,
    reconv_psf_image,
    shears1p
  )
  
  shears1m = shear + step1m 
  img1m = generate_real_mcal_image(
    gal_image,
    #psf_image,
    reconv_psf_image,
    shears1m
  ) 
  
  shears2p = shear + step2p 
  img2p = generate_real_mcal_image(
    gal_image,
    #psf_image,
    reconv_psf_image,
    shears2p
  )
  
  shears2m = shear + step2m 
  img2m = generate_real_mcal_image(
    gal_image,
    #psf_image,
    reconv_psf_image,
    shears2m
  ) 
  
  g0s = method(img0s)
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  R11 = (g1p[:,0]-g1m[:,0])/(2*step)
  R21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  R12 = (g2p[:,0]-g2m[:,0])/(2*step)
  R22 = (g2p[:,1]-g2m[:,1])/(2*step)
 
  #N. B.:The matrix is correct. 
  #The transposition will swap R12 with R21 across a batch correctly.
  R = tf.transpose(tf.convert_to_tensor(
    [[R11,R21],
     [R12,R22]],dtype=tf.float32)
  )
  
  ellip_dict = {
    'noshear':g0s,
    '1p':g1p,
    '1m':g1m,
    '2p':g2p,
    '2m':g2m,    
  } 

  return ellip_dict, R