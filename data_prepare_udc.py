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
import hashlib
data_root = '.' # Change me to store data elsewhere
os.chdir('C:\\Users\\ASUS\\Desktop\\deeplearn')

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']

test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

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
##  for folder in data_folders:
##    set_filename = folder + '.pickle'
##    dataset_names.append(set_filename)
##    if os.path.exists(set_filename) and not force:
##      # You may override by setting force=True.
##      print('%s already present - Skipping pickling.' % set_filename)
##    else:
##      print('Pickling %s.' % set_filename)
##      dataset = load_letter(folder, min_num_images_per_class)
##      try:
##        with open(set_filename, 'wb') as f:
##          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
##      except Exception as e:
##        print('Unable to save data to', set_filename, ':', e)
##  
##  return dataset_names
##
##train_datasets = maybe_pickle(train_folders, 45000)
##test_datasets = maybe_pickle(test_folders, 1800)

train_datasets=['notMNIST_large/A.pickle', 'notMNIST_large/B.pickle', 'notMNIST_large/C.pickle', 'notMNIST_large/D.pickle', 'notMNIST_large/E.pickle', 'notMNIST_large/F.pickle', 'notMNIST_large/G.pickle', 'notMNIST_large/H.pickle', 'notMNIST_large/I.pickle', 'notMNIST_large/J.pickle']
test_datasets=['notMNIST_small/A.pickle', 'notMNIST_small/B.pickle', 'notMNIST_small/C.pickle', 'notMNIST_small/D.pickle', 'notMNIST_small/E.pickle', 'notMNIST_small/F.pickle', 'notMNIST_small/G.pickle', 'notMNIST_small/H.pickle', 'notMNIST_small/I.pickle', 'notMNIST_small/J.pickle']

##def disp_number_images(data_folders):
##  for folder in data_folders:
##    pickle_filename = ''.join(folder) + '.pickle'
##    try:
##      with open(pickle_filename, 'rb') as f:
##        dataset = pickle.load(f)
##    except Exception as e:
##      print('Unable to read data from', pickle_filename, ':', e)
##      return
##    print('Number of images in ', folder, ' : ', len(dataset))
##    
##disp_number_images(train_folders)
##disp_number_images(test_folders)

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

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


























