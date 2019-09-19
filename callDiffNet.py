# Nonlinear DiffNet script to call
# Andreas Hauptmann, 2019, Oulu & UCL

import DiffNet_nonlin as DiffNet

# Data loading from Matlab files
dataSetTest  =  'data/diffNet_test_100_1e3'
dataSetTrain = 'data/diffNet_train_1000_1e3'

# Load data from matfile
dataDiffNet = DiffNet.read_data_sets(dataSetTrain,dataSetTest)

#Path for parameters and tensorboard experiment designation
netPath = 'netData/diffNet_test.ckpt'
expName = 'test'

DiffNet.trainDiffNet(netPath,expName,dataDiffNet)