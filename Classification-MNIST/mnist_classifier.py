from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import timeit


class kNNEuclideanClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def predict(self, testX):
        predY = []
        for x1 in testX:
            nearestNeigbourDist = [euclideanDistance(self.trainX[i], x1) for i in range(self.k)] 
            nearestNeigbourLabels = [self.trainY[i] for i in range(self.k)]
            for x2, y2 in zip(self.trainX, self.trainY):
                dist = euclideanDistance(x1,x2)
                if dist < max(nearestNeigbourDist):
                    maxIndex = np.argmax(nearestNeigbourDist)
                    nearestNeigbourDist[maxIndex] = dist
                    nearestNeigbourLabels[maxIndex] = y2
            predY.append(np.bincount(nearestNeigbourLabels).argmax())
        return predY
                    
def euclideanDistance(x, y):
    return np.linalg.norm(x-y)

def reshapeAndChunk(trainX, trainy, testX, testy, numTrainChunks, numTestChunks): 
    numTrainingSamples, imageSizeX, imageSizeY = trainX.shape
    numTestSamples = len(testX)
    
    trainX = trainX.reshape(numTrainingSamples, imageSizeX*imageSizeY)
    testX = testX.reshape(numTestSamples, imageSizeX*imageSizeY)
    
    trainX = np.array_split(trainX, numTrainChunks)
    trainy = np.array_split(trainy, numTrainChunks)
    testX = np.array_split(testX, numTestChunks)
    testy = np.array_split(testy, numTestChunks)

    trainingChunks, testingChunks = [], []
    for i in range(numTrainChunks):
        trainingChunks.append((trainX[i], trainy[i]))
    for j in range(numTestChunks):
        testingChunks.append((testX[j], testy[j]))
    return trainingChunks, testingChunks

def knnEuclideanClassifier(trainX, trainy, testX, numNeighbors):
    neigh = kNNEuclideanClassifier(k=numNeighbors)
    start = timeit.default_timer()
    neigh.fit(trainX, trainy)
    calcTime = timeit.default_timer() - start
    start = timeit.default_timer()
    predy = neigh.predict(testX)
    predTime = timeit.default_timer() - start
    return predy, calcTime, predTime 

def binByLabel(trainX, trainy, imageW, imageH):
    classes = dict()
    start = timeit.default_timer()
    for i in np.unique(trainy):
        classIndexes = np.nonzero(trainy == i)[0]
        classes[i] = [trainX[j].reshape(imageW*imageH) for j in classIndexes]
    binTime = timeit.default_timer() - start
    return classes, binTime

def clusterDataSet(classes, numTemplates):
    clusterLabels = []
    clusterCenters = []
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters=numTemplates, random_state=0)
    for label, datapoints in classes.items():
        kmeans.fit(datapoints)
        clusterLabels.extend([label] * numTemplates)
        clusterCenters.extend(kmeans.cluster_centers_)
    clusterTime = timeit.default_timer() - start
    return clusterCenters, clusterLabels, clusterTime

def createAndPlotConfusionMatrix(testy, predy):
    confusionMatrix = metrics.confusion_matrix(testy, predy)
    dispConfusionMatrix = metrics.ConfusionMatrixDisplay(confusionMatrix)
    dispConfusionMatrix.plot(colorbar = False)
    
def plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, correctClassifications):
    if correctClassifications:
        classifiedIndex = np.where(testy == predy)[0]
    else: 
        classifiedIndex = np.where(testy != predy)[0]
    classifiedTrue, classifiedFalse = [],[]
    for i in range(min(len(classifiedIndex), numPlotImages)):  
        classifiedImage = testX[classifiedIndex[i]]
        classifiedImage = classifiedImage.reshape(imageW,imageH)
        plt.subplot(330 + 1 + i)
        plt.title(f"True: {testy[classifiedIndex[i]]}, Pred: {predy[classifiedIndex[i]]}")
        plt.axis('off')
        plt.imshow(classifiedImage, cmap=plt.get_cmap('gray'))
        classifiedTrue.append(testy[classifiedIndex[i]])
        classifiedFalse.append(predy[classifiedIndex[i]])
    return classifiedTrue, classifiedFalse


if __name__ == '__main__':
    
    #Load the MNIST dataset
    (trainX_orig, trainy_orig), (testX_orig, testy_orig) = mnist.load_data()
    numTrainingSamples, imageW, imageH = trainX_orig.shape
    numTestingSamples = len(testX_orig)
    
    trainChunkIndex = 0
    testChunkIndex = 0
    numPlotImages = 9
    numTemplates = 64

    # Change to np.array and type to int/float
    trainX_orig = np.asarray(trainX_orig).astype(float)
    trainy_orig = np.asarray(trainy_orig).astype(int)
    testX_orig = np.asarray(testX_orig).astype(float)
    testy_orig = np.asarray(testy_orig).astype(int)
    

    ans = '-1'
    while ans != '0':
        print("""
        SELECT TASK:
        0: QUIT
        1: Task 1 kNearestNeighbour without clustering
        2: Task 2 kNearestNeighbour with k-means clustering, 64 templates per class
        """)
        ans = input("Select a task: ")
        if ans != "0":
            #Adjustable parameters
            trainingSampleSize = int(input("Training sample size: ")) #number between 0 and 60000
            testingSampleSize = int(input("Testing sample size: ")) #number between 0 and 10000
            numNeighbors = int(input("KNN, K = "))
            plotFigures = int(input("Plot classifications (0/1): "))

            assert len(trainX_orig) == len(trainy_orig)
            assert len(testX_orig) == len(testy_orig)
            assert trainingSampleSize <= numTrainingSamples
            assert testingSampleSize <= numTestingSamples
            
            numTrainChunks = numTrainingSamples//trainingSampleSize
            numTestChunks = numTestingSamples//testingSampleSize

            #Need to preprocess the data by chunking it into sizes of chunkSize datapoints in each chunk
            trainChunks, testChunks = reshapeAndChunk(trainX_orig, trainy_orig, testX_orig, testy_orig, numTrainChunks, numTestChunks)
            (trainX, trainy), (testX, testy) = trainChunks[trainChunkIndex], testChunks[testChunkIndex]
        else: 
            plotFigures = 0

        if ans == '1':
            
            #Run the kNearestNeighbor algorithm with euclidean distance
            predy, calcTime, predTime = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
            
            print("Time of fitting data: ", calcTime, " seconds")
            print("Time of prediciting data: ", predTime, " seconds")
            
            #Creating and plotting the confusion matrix
            createAndPlotConfusionMatrix(testy, predy)
            plt.title(f"({numNeighbors})NN classifier")
            
            #Creating and printing the error rate
            accRate = metrics.accuracy_score(testy, predy)
            print("Accuracy Rate of the classification: ", accRate)
            print("Error Rate of the classification: ", 1-accRate)
            plt.show()
            
        if ans == '2':
            #Cluster the training
            classes, binTime = binByLabel(trainX, trainy, imageW, imageH)     
            trainX, trainy, clusterTime = clusterDataSet(classes, numTemplates)
            print("Time of sorting data into classes: ", binTime, " seconds")
            print("Time of clustering data into ", numTemplates, " templates: ", clusterTime," seconds")
            
            #Run the kNearestNeighbor algorithm with euclidean distance
            predy, calcTime, predTime = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
            
            print("Time of fitting clustered data: ", calcTime, " seconds")
            print("Time of prediciting clustered data: ", predTime, " seconds")
            
            #Creating and plotting the confusion matrix
            createAndPlotConfusionMatrix(testy, predy)
            plt.title(f"({numNeighbors})NN classifier")
            
            #Creating and printing the error rate
            accRate = metrics.accuracy_score(testy, predy)
            print("Accuracy Rate of the classification: ", accRate)
            print("Error Rate of the classification: ", 1-accRate)
            plt.show()
            
        if plotFigures:
            #Identify and plot some of the misclassifications
            misclassifiedTrue, misclassifiedFalse = plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, False)   
            print("Actual values of misclassified digits: ", misclassifiedTrue)
            print("Predicted values of misclassified digits: ", misclassifiedFalse)    
            plt.show()
            
            #Identify and plot some of the correct classifications
            classifiedTrue, _ = plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, True)
            print("Correct classifications: ", classifiedTrue)
            plt.show()