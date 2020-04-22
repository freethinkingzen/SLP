import numpy as np
import random as rand
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class MNIST_SLP:
  def __init__(self, traincsv, testcsv):
    #settings
    self.ETA = 0.001
    self.BIAS = 1

    #clear output files
    open('train_ETA='+str(self.ETA)+'.csv', 'w').close()
    open('test_ETA='+str(self.ETA)+'.csv', 'w').close()
     
    #read data sets as NumPy arrays
    self.train_data = np.loadtxt(traincsv, dtype = float, delimiter = ',')
    self.test_data = np.loadtxt(testcsv, dtype = float, delimiter = ',')

    #capture labels from data as NumPy arrays
    self.train_labels = np.array(self.train_data[:, 0:1])
    self.test_labels = np.array(self.test_data[:, 0:1])

    #Replace targets with bias input
    self.train_data[:, 0] = self.BIAS
    self.test_data[:, 0] = self.BIAS

    #Scale data between 0 and 1
    self.train_data = np.divide(self.train_data, 255)
    self.test_data = np.divide(self.test_data, 255)

    #Initialize weight vectors to random -0.05 - 0.05
    self.weights = np.random.uniform(-0.05,0.05,(10, 785))


  def learn(self, epoch, flag):
    #read flags to set data and labels
    if(flag == "train"):
      data = self.train_data;
      labels = self.train_labels;
    if(flag == "test"):
      data = self.test_data;
      labels = self.test_labels;
    correct = 0;
    predictions = []

    #Each input in data set through perceptron
    for j in range(0,np.size(data, 0)):
      target_output = []
      activations = []
      inputs = np.array([data[j,:]])

      #initialize targets based on labels
      target_output = np.zeros(10)
      target_output[labels[j].astype('int')] = 1
      target_output = np.array([target_output])
     
      #activation function
      activations = np.dot(inputs, np.transpose(self.weights))

      #maximum value from acivation is prediction
      predictions = np.append(predictions, np.argmax(activations))

      #store activations outputs as 1 if output >0, else 0
      activations = np.where(activations>0,1,0)

      #compare labels to predictions
      if(predictions[j] == labels[j]):
        correct += 1

      #adjust weights if this is training
      if(activations[0, labels[j].astype('int')] != 1 and flag == 'train' and epoch != 0):
        a = self.ETA*np.dot(np.transpose(inputs), np.subtract(activations, target_output))
        b = np.subtract(self.weights, a.T)
        self.weights = b

    accuracy = (correct/np.size(data, 0))
    filename = str(flag)+'_ETA='+str(self.ETA)+'.csv'
    with open(filename, mode = 'a') as myfile:
      wr = csv.writer(myfile)
      wr.writerow([epoch, accuracy])

    #return predictions for confustion matrix
    if(flag == 'test'):
      return predictions

  def create_matrix(self, predictions):
    conf = confusion_matrix(self.test_labels, predictions)
    print(conf)
    plt.imshow(conf, cmap='binary', interpolation='None')
    plt.savefig('ETA='+str(self.ETA)+'con_matrix.pdf')    

EPOCHS = 100
perceptron = MNIST_SLP("mnist_train.csv","mnist_test.csv")

for i in range(0, EPOCHS):
  perceptron.learn(i, "train")
  predictions = perceptron.learn(i, "test")

perceptron.create_matrix(predictions)


