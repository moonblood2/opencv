from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import matplotlib.pyplot as plt

os.chdir('C:/Users/ASUS/Desktop/deeplearn')
image_size = 28
num_labels = 10

with open('notMNIST.pickle','rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

def reformat(data,label):
    data = data.reshape(-1,28*28)
    label = (np.arange(10)==label[:,None]).astype(np.float32)
    return data,label

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_subset = 10000
batch_size = 128
num_hidden_nodes = 1024
graph = tf.Graph()
with graph.as_default():
    
##    tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
##    tf_train_labels = tf.constant(train_labels[:train_subset])

    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,28*28))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,10))
    beta_regul = tf.placeholder(tf.float32)
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights1 = tf.Variable(tf.truncated_normal([28*28,500]))
    biases1 = tf.Variable(tf.zeros([500]))

    weights2 = tf.Variable(tf.truncated_normal([500,num_hidden_nodes]))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes]))
    
    weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes,10]))
    biases3 = tf.Variable(tf.zeros([10]))

    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset,weights1)+biases1)
    lay2_train = tf.nn.softmax(tf.matmul(lay1_train,weights2)+biases2)
    logits = tf.matmul(lay2_train,weights3)+biases3
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))+beta_regul*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
##########################################################
    train_prediction = tf.nn.softmax(logits)
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(tf.matmul(lay2_test, weights3) + biases3)
    
num_steps = 3001
regul_val = [pow(10,i) for i in np.arange(-4,-2,0.1)]
accuracy_val = []

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
##for regul in regul_val:
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):

        offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]

        feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels, beta_regul:0.00158489319246}
        
        _,l,predictions = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        if (step %500 ==0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
##        accuracy_val.append(accuracy(test_prediction.eval(), test_labels))
##print('max acc at reg =', regul_val[accuracy_val.index(max(accuracy_val))])
##plt.plot(regul_val,accuracy_val)
##plt.grid(True)
##plt.title('regul/acc')
##plt.show()
