import os
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf

os.chdir('C:/Users/ASUS/Desktop/deeplearn')

with open('notMNIST.pickle','rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('train_data',train_dataset.shape,train_labels.shape)
    print('valid_data',valid_dataset.shape,valid_labels.shape)
    print('test_data',test_dataset.shape,test_labels.shape)
    
def reformat(dataset,labels):
    dataset = dataset.reshape(-1,28*28).astype(np.float32)
    labels = (np.arange(10) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset,valid_labels = reformat(valid_dataset,valid_labels)
test_dataset,test_labels = reformat(test_dataset,test_labels)
print('Reformated')
print('train_data',train_dataset.shape,train_labels.shape)
print('valid_data',valid_dataset.shape,valid_labels.shape)
print('test_data',test_dataset.shape,test_labels.shape)
batch_size=128
graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,28*28))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,10))
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights1 = tf.Variable(tf.truncated_normal([28*28,1024]))
    biases1 = tf.Variable(tf.zeros([1024]))
    
    weights2 = tf.Variable(tf.truncated_normal([1024,10]))
    biases2 = tf.Variable(tf.zeros([10]))

    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset,weights1)+biases1)
    logits = tf.matmul(lay1_train,weights2)+biases2
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)

    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid,weights2)+biases2)

    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset,weights1)+biases1)
    test_prediction =  tf.nn.softmax(tf.matmul(lay1_test,weights2)+biases2)

def accuracy(prediction, labels):
    return 100*(np.sum(np.argmax(prediction,1)==np.argmax(labels,1))/prediction.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('TF variable Initialized')
    for step in range(3001):
        
##        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        offset = np.random.randint(0,high=train_labels.shape[0]-batch_size-1)
        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]

        feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        
        _,l,prediction = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        if(step%500 == 0):
            print('Loss at step ',step,':',l)
            print('Training accuracy:',accuracy(prediction,batch_labels))
            print('Validation accuracy:',accuracy(valid_prediction.eval(),valid_labels))
    print('***Test accuracy:',accuracy(test_prediction.eval(),test_labels))        
        
