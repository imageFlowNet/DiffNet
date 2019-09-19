# Nonlinear DiffNet script for square images
# Includes functions for loading and evaluation
# Andreas Hauptmann, 2019, Oulu & UCL

import os
import shutil
from os.path import exists
import tensorflow as tf
import numpy as np
import h5py
from random import randint

tensorboard_dir = 'ENTER PATH'

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

def default_tensorboard_dir(name):
    
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir


def summary_writers(name, expName ,cleanup=False, session=None):
    if session is None:
        session = tf.get_default_session()
    
    dname = default_tensorboard_dir(name)
    
    if cleanup and os.path.exists(dname):
        shutil.rmtree(dname)    
    
    test_summary_writer = tf.summary.FileWriter(dname + '/test_' + expName, session.graph)
    train_summary_writer = tf.summary.FileWriter(dname + '/train_' + expName)
    
    return test_summary_writer, train_summary_writer


def extract_images(filename,imageName):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  
  num_images = inData.shape[0]
  rows = inData.shape[1]
  cols = inData.shape[2]
  print(num_images, rows, cols)
  data = np.array(inData)
    
  data = data.reshape(num_images, rows, cols, 1)
  return data


class DataSet(object):

  def __init__(self, images, true):
    """Construct a DataSet"""

    assert images.shape[0] == true.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                                 true.shape))
    self._num_examples = images.shape[0]

    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1],images.shape[2])
    true = true.reshape(true.shape[0],
                            true.shape[1],true.shape[2])
    

    self._images = images
    self._true = true
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def true(self):
    return self._true

  @property
  def grad(self):
    return self._grad

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._true = self._true[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._true[start:end]


def read_data_sets(FileNameTrain,FileNameTest):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_SET = FileNameTrain
  TEST_SET  = FileNameTest
  IMAGE_NAME = 'imagesDiff'
  TRUE_NAME  = 'imagesInput'
  
  print('Start loading data')  
  train_images = extract_images(TRAIN_SET,IMAGE_NAME)
  train_true   = extract_images(TRAIN_SET,TRUE_NAME)

  
  test_images = extract_images(TEST_SET,IMAGE_NAME)
  test_true   = extract_images(TEST_SET,TRUE_NAME)


  data_sets.train = DataSet(train_images, train_true)
  data_sets.test = DataSet(test_images, test_true)

  return data_sets


def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def kappaEstimator(x_in,bSize,N):
     
    x_in = tf.reshape(x_in,[bSize,N,N,1])
    kappaEst=tf.contrib.layers.conv2d(x_in,32,3)
    kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
    kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
             
    kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)

    return kappaEst
       



def diffLayer(x_in,bSize,N,layNum):
    
    zeroNHorz=tf.constant(0.0, shape=[bSize,N,1,1])
    zeroNVert=tf.constant(0.0, shape=[bSize,1,N,1])
    
    dt=tf.constant(0.2, shape=[1])
    dt = tf.Variable(dt,name='dt_' + str(layNum))
    
    x_update=tf.reshape(x_in ,[bSize,N,N,1])

    kappa=kappaEstimator(x_in,bSize,N)
    kapDiag  = tf.reshape(kappa[:,:,:,0],[bSize,N,N,1])
    kapUp    = tf.reshape(kappa[:,:,:,1],[bSize,N,N,1])
    kapDown  = tf.reshape(kappa[:,:,:,2],[bSize,N,N,1])
    kapLeft  = tf.reshape(kappa[:,:,:,3],[bSize,N,N,1])
    kapRight = tf.reshape(kappa[:,:,:,4],[bSize,N,N,1])
    
    xUp = tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)
    xDown = tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)
    xLeft = tf.concat([zeroNHorz,x_update[:,:,0:N-1]],axis=2)
    xRight = tf.concat([x_update[:,:,1:N],zeroNHorz],axis=2)
        
    xDiag  = tf.multiply(kapDiag,x_update)
    xUp    = tf.multiply(kapUp,xUp)
    xDown  = tf.multiply(kapDown,xDown)
    xLeft  = tf.multiply(kapLeft,xLeft)
    xRight = tf.multiply(kapRight,xRight)
        
    x_update = x_update + dt*(xUp + xDown + xLeft + xRight - xDiag) #Test +xDiag
    
    return x_update, kappa
    

def trainDiffNet(netPath,expName,dataDiffNet,
                 bSize=int(16),
                 N=int(96),
                 layerNum=int(5),
                 trainIter=int(50001),
                 tensorboardFL = False,
                 LR_init=4e-3
                 ):
     
    print('--------------------> DiffNet Train Init <--------------------')
    

    sess = tf.InteractiveSession()    
      
    imSize=dataDiffNet.train.true.shape
    N=imSize[1]  

    # Create the placeholder
    imag = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])
    true = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])
    
    #Diff Net    
    with tf.name_scope('DiffNet'):
      
      x_update=tf.reshape(imag ,[bSize,N*N])
      
      
      for iii in range(layerNum):
          x_update, kappaEst = diffLayer(x_update,bSize,N,iii)
        
      x_update=tf.reshape(x_update,[bSize,N,N])
      y_diff = tf.nn.relu(x_update)
    
      
    
    saver = tf.train.Saver()
    
    
    with tf.name_scope('optimizer'):
         
         loss = tf.norm(tf.subtract(tf.nn.relu(true),y_diff))/float(bSize)
         rel_loss = tf.norm(tf.subtract(tf.nn.relu(true),y_diff))/tf.norm(tf.nn.relu(true))
         learningRate=tf.constant(1e-3)
         train_step = tf.train.AdamOptimizer(learningRate).minimize(loss)
    
    if(tensorboardFL):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('rel_loss', rel_loss)
            tf.summary.scalar('psnr', psnr(y_diff, true))
        
        
            tf.summary.image('result', tf.reshape(y_diff[1],[1, imSize[1], imSize[2], 1]) )
            tf.summary.image('kappaEst', tf.reshape(kappaEst[1,:,:,0],[1, imSize[1], imSize[2], 1]) )
            tf.summary.image('true', tf.reshape(true[1],[1, imSize[1], imSize[2], 1]) )
            tf.summary.image('imag', tf.reshape(imag[1],[1, imSize[1], imSize[2], 1]) )
            tf.summary.image('diff', tf.reshape(true[1]-y_diff[1],[1, imSize[1], imSize[2], 1]) )
            
            merged_summary = tf.summary.merge_all()
            expName='DiffNet_' + expName
            test_summary_writer, train_summary_writer = summary_writers(name, expName ,cleanup=False)
            
    
    
    sess.run(tf.global_variables_initializer())
    
    feed_test={imag: dataDiffNet.test.images[0:bSize],
                 true: dataDiffNet.test.true[0:bSize]}
                 
    

    lVal = LR_init
    for i in range(trainIter):
          
          batch = dataDiffNet.train.next_batch(bSize)   
          feed_train={imag: batch[0], true: batch[1], learningRate: lVal}
          
          if(tensorboardFL):
              _, merged_summary_result_train = sess.run([train_step, merged_summary],
                                          feed_dict=feed_train)
          else:
              sess.run([train_step],feed_dict=feed_train)
          
            
          if i % 10000 == 0:
              lVal=lVal/2.0
          
          if i % 20 == 0:
                        
            tBeg = randint(0,9900)
            tEnd= tBeg+bSize
            
            feed_test={imag: dataDiffNet.test.images[tBeg:tEnd],
                         true: dataDiffNet.test.true[tBeg:tEnd]}
            
            if(tensorboardFL):
                loss_result, rel_loss_res, merged_summary_result = sess.run([loss, rel_loss, merged_summary],
                              feed_dict=feed_test)
        
                train_summary_writer.add_summary(merged_summary_result_train, i)
                test_summary_writer.add_summary(merged_summary_result, i)
            else:
                loss_result, rel_loss_res = sess.run([loss, rel_loss],feed_dict=feed_test)
        
                
            print('iter={}, loss={}, rel.loss={}'.format(i, loss_result,rel_loss_res))
            
    
       
        
    save_path = saver.save(sess, filePath)
    print("Model saved in file: %s" % save_path)
        
        
    
    print('--------------------> DONE <--------------------')
        









