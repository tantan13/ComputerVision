#!/usr/bin/python3

import numpy as np
import math

def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = k // 2
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors drawn from 1D Gaussian distributions.
  """

  ############################
  ### TODO: YOUR CODE HERE ###

  k = cutoff_frequency * 4 + 1
  mean = math.floor(k/2)
  sd = cutoff_frequency
    
  values = [i for i in range (k)] #values list from 0 to K
  for i,x in enumerate(values):
       values[i] = (1/(math.sqrt(2*math.pi)*sd)) * (math.exp(-(1/(2*sd*sd))*(x - mean)**2))
    
  kernel = np.outer(values,values) #gets outer product 
  num = sum(sum(kernel)) #sum the row and columns
  for r in range(len(kernel)):
      for c in range(len(kernel)):
         kernel[r][c] = kernel[r][c]/num
  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filtered_image = np.zeros(image.shape)
 
  pad_width = int(filter.shape[0]/2)
  pad_height = int(filter.shape[1]/2)
    
  image = np.pad(image,((pad_width, pad_width), (pad_height, pad_height), (0,0)), 'constant')
  
  for k in range(image.shape[2]):
      for i in range(pad_width, pad_width + filtered_image.shape[0]):
          for j in range(pad_height, pad_height + filtered_image.shape[1]):

               filtered_image[i - pad_width][j - pad_height][k] = np.dot(filter.reshape(-1), (image[i - pad_width : i + pad_width + 1, j - pad_height : j + pad_height + 1, k]).reshape(-1)) #doing filter on the neighborhood for each center
  
  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

#   raise NotImplementedError('`create_hybrid_image` function in ' +
#     '`student_code.py` needs to be implemented')

  low_frequencies = my_imfilter(image1, filter)
  low_f_content = my_imfilter(image2, filter)
  high_frequencies = image2 - low_f_content
  hybrid_image = low_frequencies + high_frequencies
  hybrid_image = np.clip(hybrid_image, 0, 1)
  #print(hybrid_image)
#   ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
