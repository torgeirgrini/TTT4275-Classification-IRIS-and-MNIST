import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.datasets import load_iris





def splitDataSet(data, splitIndex, classSize, trainingFirst, C, D):
    classSetosa = data[0:classSize]
    classVersicolour = data[classSize:(C-1)*classSize]
    classVirginica = data[(C-1)*classSize:C*classSize]
    
    trainingSetSetosa = classSetosa[0:splitIndex]
    trainingSetVersicolour = classVersicolour[0:splitIndex]
    trainingSetVirginica = classVirginica[0:splitIndex]
    trainingSet = np.concatenate((trainingSetSetosa, trainingSetVersicolour, trainingSetVirginica))

    testSetSetosa = classSetosa[splitIndex:]
    testSetVersicolour = classVersicolour[splitIndex:]
    testSetVirginica = classVirginica[splitIndex:]
    testSet = np.concatenate((testSetSetosa, testSetVersicolour, testSetVirginica))

    if not(trainingFirst):            #swicth the sets if we want training the training set to be the last part
        temp = testSet
        testSet = trainingSet
        trainingSet = temp

    
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

#Descision rule linear discriminant classifier
def descisionRule(sample,W):
    classIndex = np.argmax(g(sample,W))
    return classIndex

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
    return W, predictedLabels


def cheapClassifierTest(W, testSamples):
    #### CHEAP TEST OF CLASSIFIER
    counter = 0
    for x in testSamples:
        if counter == 0:
            print("#################### TEST CLASS 1 ##################")
        elif counter  == 20:
            print("#################### TEST CLASS 2 #################")
        elif counter == 40:
            print("#################### TEST CLASS 3 #################")
        print("Sample: ", x)
        print("Function value: ",g(x,W))
        print("Class: ",descisionRule(x,W))
        print("\n")
        counter += 1

################# OTHER STUFF

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

############ HISTOGRAM OF FEATURES


###
       

def main():
    irisDataSet = load_iris()
    ans = '-1'
    while ans != '0':
        print("""
        SELECT TASK:
        0. QUIT
        1. Task 1c: 30 first for training, 20 last for test
        2. Task 1d: 30 last for training, 20 first for test
        3. Task 2a: HISTOGRAM PLOT
        4. Task 2a: REMOVES 2nd FEATURE FROM DATASET
        """)
        ans = input("Select a task: ")
        if ans == '1':

            C = 3 # number of classes
            D = 4 #number of features 

            trainingSet, testSet = splitDataSet(irisDataSet['data'][0:], 30, 50, True, C, D)
            # trueLabelsTrainingSet, samples = initTargetsAndTrainingSet()
            trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, 30, C)
            trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, 20, C)
            
            # INIT redefined W matrix
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])                                      #total matrix/ weight matrix/redefined W from chap 3.2
            
            
            # Training linear classifier on training set
            alpha = 0.01
            W, predictedLabelsTrainingSet = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, 1000, C, D)

            # Testing classifier on test set
            # cheapClassifierTest(W, testSamples)

            # Error rate on training set
            predictedLabelsTrainingSet = roundPredictedLabels(predictedLabelsTrainingSet)
            errorRateTrainingSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTrainingSet, predictedLabelsTrainingSet)
            print("ERROR RATE TRAINING SET: ",errorRateTrainingSet)

            # Confusion matrix on training set
            confusionMatrixTrainingSet = getConfusionMatrix(trueLabelsTrainingSet,predictedLabelsTrainingSet, C)
            print(confusionMatrixTrainingSet)
        
            # Training linear classifier on test set, error rate on test set
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])  
            _, predLabelsTestSet = trainLinearClassifier(trueLabelsTestSet, testSamples, W, alpha, 1000, C, D)
            predLabelsTestSet = roundPredictedLabels(predLabelsTestSet)
            errorRateTestSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTestSet, predLabelsTestSet)
            print("ERROR RATE TEST SET: ", errorRateTestSet)

            # Confusion matrix on test set
            confusionMatrixTestSet = getConfusionMatrix(trueLabelsTestSet,predLabelsTestSet, C)
            print(confusionMatrixTestSet)
            



        elif ans == '2':
            C = 3 # number of classes
            D = 4 #number of features 

            trainingSet, testSet = splitDataSet(irisDataSet['data'][0:], 20, 50, False, C, D)
            # trueLabelsTrainingSet, samples = initTargetsAndTrainingSet()
            trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, 30, C)
            trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, 20, C)
            
            # INIT redefined W matrix
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])                                      #total matrix/ weight matrix/redefined W from chap 3.2
            
            # Training linear classifier on training set
            alpha = 0.01
            W, predictedLabelsTrainingSet = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, 1000, C, D)

            # Testing classifier on test set
            # cheapClassifierTest(W, testSamples)

            # Error rate on training set
            predictedLabelsTrainingSet = roundPredictedLabels(predictedLabelsTrainingSet)
            errorRateTrainingSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTrainingSet, predictedLabelsTrainingSet)
            print("ERROR RATE TRAINING SET: ",errorRateTrainingSet)

            # Confusion matrix on training set
            confusionMatrixTrainingSet = getConfusionMatrix(trueLabelsTrainingSet,predictedLabelsTrainingSet, C)
            print(confusionMatrixTrainingSet)
            
            # Training linear classifier on test set, error rate on test set
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])  
            _, predLabelsTestSet = trainLinearClassifier(trueLabelsTestSet, testSamples, W, alpha, 1000, C, D)
            predLabelsTestSet = roundPredictedLabels(predLabelsTestSet)
            errorRateTestSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTestSet, predLabelsTestSet)
            print("ERROR RATE TEST SET: ", errorRateTestSet)

            # Confusion matrix on test set
            confusionMatrixTestSet = getConfusionMatrix(trueLabelsTestSet,predLabelsTestSet, C)
            print(confusionMatrixTestSet)
        elif ans == "3":
            fig, axes = plt.subplots(nrows= 2, ncols=2)
            colors= ['blue', 'red', 'green']

            for i, ax in enumerate(axes.flat):
                for label, color in zip(range(len(irisDataSet.target_names)), colors):
                    ax.hist(irisDataSet.data[irisDataSet.target==label, i], label=             
                                        irisDataSet.target_names[label], color=color)
                    ax.set_xlabel(irisDataSet.feature_names[i])  
                    ax.legend(loc='upper right')
            plt.show()
        elif ans == "4":
            ## second feature has the most overlap
            ###### ADD FUNCTIONALITY TO REMOVE FEATURES AS YOU WISH
            
            C = 3 # number of classes
            D = 3 #number of features 


            ## REMOVE FEATURE
            irisDataSetReduced = irisDataSet['data'][0:]
            irisDataSetReduced = np.delete(irisDataSetReduced,1, 1)         #np.s_[1:1], 1)
            print(irisDataSetReduced[0])

            trainingSet, testSet = splitDataSet(irisDataSetReduced, 30, 50, True, C, D)
            trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, 30, C)
            trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, 20, C)
            
            # INIT redefined W matrix
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])                                      #total matrix/ weight matrix/redefined W from chap 3.2
            
            
            # Training linear classifier on training set
            alpha = 0.01
            W, predictedLabelsTrainingSet = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, 1000, C, D)

            # Testing classifier on test set
            # cheapClassifierTest(W, testSamples)

            # Error rate on training set
            predictedLabelsTrainingSet = roundPredictedLabels(predictedLabelsTrainingSet)
            errorRateTrainingSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTrainingSet, predictedLabelsTrainingSet)
            print("ERROR RATE TRAINING SET: ",errorRateTrainingSet)

            # Confusion matrix on training set
            confusionMatrixTrainingSet = getConfusionMatrix(trueLabelsTrainingSet,predictedLabelsTrainingSet, C)
            print(confusionMatrixTrainingSet)
        
            # Training linear classifier on test set, error rate on test set
            W = np.zeros((C,D))                                     
            w0 = np.zeros((C,1))
            W = np.block([W, w0])  
            _, predLabelsTestSet = trainLinearClassifier(trueLabelsTestSet, testSamples, W, alpha, 1000, C, D)
            predLabelsTestSet = roundPredictedLabels(predLabelsTestSet)
            errorRateTestSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTestSet, predLabelsTestSet)
            print("ERROR RATE TEST SET: ", errorRateTestSet)

            # Confusion matrix on test set
            confusionMatrixTestSet = getConfusionMatrix(trueLabelsTestSet,predLabelsTestSet, C)
            print(confusionMatrixTestSet)
            




if __name__ == "__main__":
    main()


