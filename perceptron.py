import numpy as np
import random as rand
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def perceptron(epoch, data, labels, weights, ETA, flag):
  correct = 0;
  predictions = []
  for j in range(0,np.size(data, 0)):
    target_output = []
    activations = []
    inputs = np.array([data[j,:]])

    target_output = np.zeros(10)
    target_output[labels[j].astype('int')] = 1
    target_output = np.array([target_output])
   
    #activation function
    activations = np.dot(inputs, np.transpose(weights))

    #maximum value from acivation is prediction
    predictions = np.append(predictions, np.argmax(activations))

    #store activations as 1 if >0, else 0
    activations = np.where(activations>0,1,0)

    if(predictions[j] == labels[j]):
      correct += 1;

    if(activations[0, labels[j].astype('int')] != 1 and flag == 'train'):
      a = ETA*np.dot(np.transpose(inputs), np.subtract(activations, target_output))
      b = np.subtract(weights, a.T)
      weights = b

    if(j == 0 and flag == 'train'):
      accuracy = (correct/10)
      filename = str(flag)+'_ETA='+str(ETA)+'.csv'
      with open(filename, mode = 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow([-1, accuracy])
    
  accuracy = (correct/np.size(data, 0))

  filename = str(flag)+'_ETA='+str(ETA)+'.csv'
  with open(filename, mode = 'a') as myfile:
    wr = csv.writer(myfile)
    wr.writerow([epoch, accuracy])

  if(flag == 'train'):
    return weights
  if(flag == 'test'):
    return predictions


  
#Constants
ETA = 0.001
EPOCHS = 1
OUTPUT_NEURONS = 10

#clear output file for new run
open('test_ETA='+str(ETA)+'.csv', 'w').close()

#load data
train_data = np.loadtxt('mnist_train.csv', dtype = float, delimiter = ',')
test_data = np.loadtxt('mnist_test.csv', dtype = float, delimiter = ',')


#Initialize weight vectors to random
weights = np.random.uniform(-0.05,0.05,(OUTPUT_NEURONS,np.size(train_data,1)))

#Capture targets from data
train_labels = np.array(train_data[:, 0:1])
test_labels = np.array(test_data[:, 0:1])

#Replace targets with bias input
train_data[:, 0] = 1
test_data[:, 0] = 1

#Scale data
train_data = np.divide(train_data, 255)
test_data = np.divide(test_data, 255)

for i in range(0, EPOCHS):

  weights = perceptron(i, train_data, train_labels, weights, ETA, 'train')

  predictions = perceptron(i, test_data, test_labels, weights, ETA, 'test')

  if(i == EPOCHS-1):
    conf = confusion_matrix(test_labels, predictions)
    print(conf)
    plt.imshow(conf, cmap='binary', interpolation='None')
    plt.savefig('ETA='+str(ETA)+'con_matrix.pdf')    
