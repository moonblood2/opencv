from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random

os.chdir('C:\\Users\\ASUS\\Desktop\\deeplearn')

train_folder = 'notMNIST_large'

test_folder = 'notMNIST_small'
'''
os.chdir(os.path.join(os.getcwd(),test_folder))

for f_ABC in os.listdir():
    path = os.path.join(os.getcwd(),f_ABC)
    rand = np.random.randint(0,high=len(os.listdir(path)))
    im_path = os.path.join(os.getcwd(),f_ABC,os.listdir(path)[rand])
    try:
        im = Image.open(im_path)
        im.show()
    except IOError as e:
        print('e=',e)
'''
##
##image_size = 28  # Pixel width and height.
##pixel_depth = 255.0  # Number of levels per pixel.
##
##def load_letter(folder, min_num_images):
##  """Load the data for a single letter label."""
##  image_files = os.listdir(folder)
##  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
##                         dtype=np.float32)
##  print(folder)
##  num_images = 0
##  for image in image_files:
##    image_file = os.path.join(folder, image)
##    try:
##      image_data = (ndimage.imread(image_file).astype(float) - 
##                    pixel_depth / 2) / pixel_depth
##      if image_data.shape != (image_size, image_size):
##        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
##      dataset[num_images, :, :] = image_data
##      num_images = num_images + 1
##    except IOError as e:
##      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
##    
##  dataset = dataset[0:num_images, :, :]
##  if num_images < min_num_images:
##    raise Exception('Many fewer images than expected: %d < %d' %
##                    (num_images, min_num_images))
##    
##  print('Full dataset tensor:', dataset.shape)
##  print('Mean:', np.mean(dataset))
##  print('Standard deviation:', np.std(dataset))
##  return dataset
##        
##def maybe_pickle(data_folders, min_num_images_per_class, force=False):
##  dataset_names = []
##  for folder in os.listdir(data_folders):
##    set_filename = folder + '.pickle'
##    dataset_names.append(set_filename)
##    if os.path.exists(set_filename) and not force:
##      # You may override by setting force=True.
##      print('%s already present - Skipping pickling.' % set_filename)
##    else:
##      print('Pickling %s.' % set_filename)
##      
##      dataset = load_letter(os.path.join(train_folders,folder), min_num_images_per_class)
##      try:
##        with open(set_filename, 'wb') as f:
##          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
##      except Exception as e:
##        print('Unable to save data to', set_filename, ':', e)
##  
##  return dataset_names
##train_folders = os.path.join(os.getcwd(),train_folder)
####train_datasets = maybe_pickle(train_folders, 45000)
##train_datasets = maybe_pickle(train_folders, 1800)
##
##
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_folders=os.path.join(os.getcwd(),'notMNIST_large')
test_folders=os.path.join(os.getcwd(),'notMNIST_small')
train_datasets=os.path.join(os.getcwd(),'train_pickle')
test_datasets=os.path.join(os.getcwd(),'test_pickle')

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
