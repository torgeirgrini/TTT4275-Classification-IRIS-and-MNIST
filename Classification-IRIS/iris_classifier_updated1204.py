import enum
from unittest.util import _count_diff_all_purpose
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.datasets import load_iris

def splitDataSet(data, splitIndex, classSize, trainingPrior, C, D):
    classSetosa = data[0:classSize]
    classVersicolour = data[classSize:(C-1)*classSize]
    classVirginica = data[(C-1)*classSize:C*classSize]
    
    firstSetSetosa = classSetosa[0:splitIndex]
    firstSetVersicolour = classVersicolour[0:splitIndex]
    firstSetVirginica = classVirginica[0:splitIndex]
    firstSet = np.concatenate((firstSetSetosa, firstSetVersicolour, firstSetVirginica))

    lastSetSetosa = classSetosa[splitIndex:]
    lastSetVersicolour = classVersicolour[splitIndex:]
    lastSetVirginica = classVirginica[splitIndex:]
    lastSet = np.concatenate((lastSetSetosa, lastSetVersicolour, lastSetVirginica))

    if trainingPrior:            #swicth the sets if we want training the training set to be the last part
        trainingSet = firstSet
        testSet = lastSet
    else:
        trainingSet = lastSet
        testSet = firstSet

    return trainingSet, testSet

def initTrueLabelsAndSamples(data, classSize, C):
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    samples = [] 
    trueLabels  = [] #targets
    counter = 0
    for x in data:
        samples.append(np.append(x,1)) # redefinition from chap 3.2
        if counter < classSize:
            trueLabels.append(e1)
        elif counter < (C-1)*classSize:
            trueLabels.append(e2)
        elif counter < (C)*classSize:
            trueLabels.append(e3)
        counter += 1
    trueLabels = np.array(trueLabels)
    samples = np.array(samples) 
    return trueLabels, samples

############################ CLASSIFIER #############################
#Linear discriminant classifier function
def g(x,W):
    return np.matmul(W,x)

def sigmoid(sample,W):
    return (1/(1+np.exp(-g(sample,W)))) #outputs g_ik


def classPredictions(X,W):
    predictedLabels = []
    for x_i in X:
        predictedLabels.append(sigmoid(x_i, W))
    return np.array(predictedLabels)

# Implementation of 3.22, 3.23
def nextW(predictedLabels, trueLabels, samples, prevW, alpha, C, D):
    grad_W_MSE = np.zeros((C,D+1))
    for k in range(len(samples)):
        grad_gk_MSE = predictedLabels[k]-trueLabels[k]
        grad_zk_g = np.multiply(predictedLabels[k],1-predictedLabels[k])
        grad_W_zk = np.reshape(samples[k],(1,D+1))
        grad_W_MSE +=  np.matmul(np.reshape(np.multiply(grad_gk_MSE, grad_zk_g),(C,1)), grad_W_zk)

    return prevW - alpha*grad_W_MSE

def trainLinearClassifier(targets, samples, W_0, alpha, iterations, C, D):
    W = W_0
    for m in range(iterations):
        predictedLabels = classPredictions(samples,W)
        W = nextW(predictedLabels, targets, samples, W, alpha, C, D)
    return predictedLabels

##################

def roundPredictedLabels(predictedLabels):
    for tk in predictedLabels:
        classIndex = np.argmax(tk)
        tk[classIndex] = 1
        for i in range(len(tk)):
            if i != classIndex:
                tk[i] = 0
    return predictedLabels


def getConfusionMatrix(trueLabels ,predictedLabels, C):
    confusionMatrix = np.zeros((C,C))
    for k in range(len(trueLabels)):
        trueClassIndex = np.argmax(trueLabels[k])
        predClassindex = np.argmax(predictedLabels[k])
        confusionMatrix[trueClassIndex][predClassindex] += 1
    return confusionMatrix


def main():
    irisDataSet = load_iris()
    

    print("""
    LINEAR CLASSIFIER FOR THE IRIS DATA SET""")
    ans = '-1'
    while ans != '0':
        C = 3      # number of classes
        D = 4      # number of features
        classSize = 50

        print("""
(0) QUIT
(1) CONFIGURE LINEAR CLASSIFIER
(2) PLOT HISTOGRAM OF FEATURES""")
        ans = input("SELECT: ")
        if ans == '1':
            print("CONFIGURE LINEAR CLASSIFIER")
            print("""Which feature(s) do you want to remove?
(0) Sepal length
(1) Sepal width
(2) Petal length
(3) Petal width
(any char) Keep all features """)
            featureRemoval = input("Enter features as list (ex. 1 | ex. 2, 4| ex. K): ")

            removedFeatureIndexes = []
            for c in featureRemoval:
                if c.isdigit():
                    removedFeatureIndexes.append(int(c))
            removedFeatureIndexes.sort()

            if len(removedFeatureIndexes) > 0:
                D = D-len(removedFeatureIndexes)
                dataSet = irisDataSet['data'][0:]
                dataSet = np.delete(dataSet,removedFeatureIndexes, 1)
                print("REMOVED FEATURES: ", removedFeatureIndexes)
            else:
                dataSet = irisDataSet['data'][0:]
                print("KEPT ALL FEATURES")
            

            splitIndex = int(input("Choose split index between training set and test set of each class (0-50): "))
            trainingPrior = bool(int(input(f"""(0) Training set POST to index {splitIndex} \n(1) Training set PRIOR to index {splitIndex} \nSELECT: """)))
            trainingSet, testSet = splitDataSet(dataSet, splitIndex, classSize, trainingPrior, C, D)
            if trainingPrior:
                trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, splitIndex, C)
                trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, classSize-splitIndex, C)
            else:
                trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, classSize-splitIndex, C)
                trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, splitIndex, C)
            # INIT W matrix
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])                                      #total matrix/ weight matrix/redefined W from chap 3.2
            
            # Training linear classifier on training set
            alpha = 0.005
            iterations = 400
            predictedLabelsTrainingSet = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, iterations, C, D)

            predictedLabelsTrainingSet = roundPredictedLabels(predictedLabelsTrainingSet)
            errorRateTrainingSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTrainingSet, predictedLabelsTrainingSet)
            print("ERROR RATE TRAINING SET: ",errorRateTrainingSet)

            confusionMatrixTrainingSet = getConfusionMatrix(trueLabelsTrainingSet,predictedLabelsTrainingSet, C)
            print(confusionMatrixTrainingSet)
            dispConfusionMatrixTrainingSet = sklearn.metrics.ConfusionMatrixDisplay(confusionMatrixTrainingSet, display_labels=irisDataSet.target_names)
            dispConfusionMatrixTrainingSet.plot()
            plt.title("Confusion Matrix Training Set")
            plt.show()
            
            # Training linear classifier on test set, error rate on test set
            predLabelsTestSet = trainLinearClassifier(trueLabelsTestSet, testSamples, W, alpha, iterations, C, D)
            predLabelsTestSet = roundPredictedLabels(predLabelsTestSet)
            errorRateTestSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTestSet, predLabelsTestSet)
            print("ERROR RATE TEST SET: ", errorRateTestSet)

            confusionMatrixTestSet = getConfusionMatrix(trueLabelsTestSet,predLabelsTestSet, C)
            print(confusionMatrixTestSet)
            dispConfusionMatrixTestSet = sklearn.metrics.ConfusionMatrixDisplay(confusionMatrixTestSet, display_labels=irisDataSet.target_names)
            dispConfusionMatrixTestSet.plot()
            plt.title("Confusion Matrix Test Set")
            plt.show()
            
        elif ans == '2':
            print("HISTOGRAM PLOT")
            _, axes = plt.subplots(nrows= 2, ncols=2)
            colors= ['gold', 'darkorchid', 'red']

            for i, ax in enumerate(axes.flat):
                for label, color in zip(range(len(irisDataSet.target_names)), colors):
                    ax.hist(irisDataSet.data[irisDataSet.target==label, i], label=             
                                        irisDataSet.target_names[label], color=color)
                    ax.set_xlabel(irisDataSet.feature_names[i])  
                    ax.legend(loc='upper right')
            plt.show()
        

if __name__ == "__main__":
    main()


