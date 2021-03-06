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
    dataCopy = data
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    samples = [] 
    trueLabels  = [] #targets
    counter = 0
    for x in dataCopy:
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
    errorRates = []
    MSEValues = []
    for m in range(iterations):
        predictedLabels = classPredictions(samples,W)
        W = nextW(predictedLabels, targets, samples, W, alpha, C, D)
        if m % 50 == 0:
            MSEValues.append(calculateMSE(targets, predictedLabels))
            roundedPredictedLabels = roundPredictedLabels(predictedLabels)
            errorRates.append(1 - sklearn.metrics.accuracy_score(targets, roundedPredictedLabels))
    return W, predictedLabels, errorRates, MSEValues

def classifyWithTrainedClassifier(samples, W):
    predictedLabels = classPredictions(samples,W)
    predictedLabels = roundPredictedLabels(predictedLabels)
    return predictedLabels

##################

def roundPredictedLabels(predictedLabels):
    predCopy = np.copy(predictedLabels)
    for tk in predCopy:
        classIndex = np.argmax(tk)          #Use of decision rule
        tk[classIndex] = 1
        for i in range(len(tk)):
            if i != classIndex:
                tk[i] = 0
    return predCopy


def getConfusionMatrix(trueLabels ,predictedLabels, C):
    confusionMatrix = np.zeros((C,C))
    for k in range(len(trueLabels)):
        trueClassIndex = np.argmax(trueLabels[k])
        predClassindex = np.argmax(predictedLabels[k])
        confusionMatrix[trueClassIndex][predClassindex] += 1
    return confusionMatrix

def calculateMSE(trueLabels, predLabels):
    MSE = 0
    for k in range(len(trueLabels)):
        MSE += np.matmul(np.transpose((predLabels[k] - trueLabels[k])), (predLabels[k] - trueLabels[k]))
    return (1/2)*MSE



def removeFeatures(dataSet, featureIndexes,D):
    removedFeatureIndexes = []
    for c in featureIndexes:
        if c.isdigit():
            removedFeatureIndexes.append(int(c))
    removedFeatureIndexes.sort()

    if len(removedFeatureIndexes) > 0:
        D = D-len(removedFeatureIndexes)
        newDataSet = dataSet['data'][0:]
        newDataSet = np.delete(newDataSet,removedFeatureIndexes, 1)
        print("REMOVED FEATURES: ", removedFeatureIndexes)
    else:
        newDataSet = dataSet['data'][0:]
        print("KEPT ALL FEATURES")
    return newDataSet, D

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
        (2) PLOT HISTOGRAM OF FEATURES
        (3) ALPHA DECISION PLOT""")
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

            dataSet, numFeatures = removeFeatures(irisDataSet, featureRemoval,D)
            D = numFeatures

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
            alpha = 0.006
            iterations = 2000
            W ,predictedLabelsTrainingSet, errorRatesTraining, MSEValues = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, iterations, C, D)
            print("ERROR RATE TRAINING SET: ",errorRatesTraining[-1])

            confusionMatrixTrainingSet = getConfusionMatrix(trueLabelsTrainingSet,predictedLabelsTrainingSet, C)
            print(confusionMatrixTrainingSet)
            dispConfusionMatrixTrainingSet = sklearn.metrics.ConfusionMatrixDisplay(confusionMatrixTrainingSet, display_labels=irisDataSet.target_names)
            dispConfusionMatrixTrainingSet.plot()
            plt.title("Confusion Matrix Training Set")
            
            # Test trained classifier on test set
            predLabelsTestSet = classifyWithTrainedClassifier(testSamples, W)
            errorRateTestSet = 1 - sklearn.metrics.accuracy_score(trueLabelsTestSet, predLabelsTestSet)
            print("ERROR RATE TEST SET: ", errorRateTestSet)

            confusionMatrixTestSet = getConfusionMatrix(trueLabelsTestSet,predLabelsTestSet, C)
            print(confusionMatrixTestSet)
            dispConfusionMatrixTestSet = sklearn.metrics.ConfusionMatrixDisplay(confusionMatrixTestSet, display_labels=irisDataSet.target_names)
            dispConfusionMatrixTestSet.plot()
            plt.title("Confusion Matrix Test Set")
            plt.show()

            xAxisIter = np.arange(0,iterations,50)
            plt.plot(xAxisIter ,errorRatesTraining)
            plt.title("Error Rate Training Set")
            plt.xlabel("Iterations")
            plt.ylabel("Error Rate")
            plt.show()
            

            plt.plot(xAxisIter, MSEValues)
            plt.title("MSE Values")
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
        elif ans == '3':
            print("ALPHA DECISION PLOT")
            splitIndex = 30
            trainingSet, testSet = splitDataSet(irisDataSet['data'][0:], splitIndex, classSize, True, C, D)
            trueLabelsTrainingSet, trainingSamples = initTrueLabelsAndSamples(trainingSet, splitIndex, C)
            trueLabelsTestSet, testSamples = initTrueLabelsAndSamples(testSet, classSize-splitIndex, C)
            
            # Training linear classifier on training set
            alphas = [0.0001, 0.001, 0.005, 0.01, 0.05]
            iterations = 4000
            xAxisIter = np.arange(0,iterations,50)
            figure, axis = plt.subplots(2,1)
            axis[0].set_title("MSE Values for Different \u03B1")
            axis[0].set_xlabel("Iterations")
            axis[0].set_ylabel("MSE")

            axis[1].set_title("Error Rates for Different \u03B1")
            axis[1].set_xlabel("Iterations")
            axis[1].set_ylabel("Error Rate")
            for alpha in alphas:
                W = np.zeros((C,D))                                     
                w0 = np.zeros((C,1))
                W = np.block([W, w0])   
                W ,predictedLabelsTrainingSet, errorRatesTraining, MSEValues = trainLinearClassifier(trueLabelsTrainingSet, trainingSamples, W, alpha, iterations, C, D)
                axis[0].plot(xAxisIter, MSEValues, label='\u03B1' + " = " + str(alpha))
                axis[0].legend(loc="upper right")
                axis[1].plot(xAxisIter, errorRatesTraining, label='\u03B1' + " = " + str(alpha))
                axis[1].legend(loc="upper right")
            
            plt.show()
        

        
if __name__ == "__main__":
    main()


