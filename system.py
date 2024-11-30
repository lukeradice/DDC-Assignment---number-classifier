from utils import *
import numpy as np


def image_to_reduced_feature(images, split='train'):
    # tenBestItOne = [129, 438, 128, 98, 411, 652, 684, 713, 127, 221]
    # tenBestItTwo = [128, 130, 438, 129, 411, 684, 127, 683, 713, 98]
    return images


def training_model(train_feature_vectors, train_labels):
    
    return ImprovedModel(train_feature_vectors)
    # return NullModel()
    # null model takes in the training data and has a method predict
    # which returns an estimated class label for each row that is input

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_feature_vectors):
        self.train_feature_vectors = train_feature_vectors
        self.train, self.train_labels = get_dataset('train')
        # print(train_features.shape)
        # print(calculateBestFeatures(train_features, train_labels))

    def predict(self, test):
        print("whatttttttttttt")
        # features = [129, 438, 128, 98, 411, 652, 684, 713, 127, 221]
            
        # Select the desired features from the training and test data
        # train = self.train[:, features]
        # test = test[:, features]
        train = self.train
        print(test.shape)
        print(train.shape)

        # nearest neighbour
        trainTestDotProd = np.dot(
                                 test, train.transpose())
        modtest = np.sqrt(np.sum(test * test, axis=1))
        modtrain = np.sqrt(
                      np.sum(train * train, axis=1))
        outerProd = np.maximum(np.outer(modtest, modtrain.transpose()), 0.000000001)
        distance = trainTestDotProd / outerProd

        # cosine distance
        nearest = np.argmax(distance, axis=1)
        mdist = np.max(distance, axis=1)
        return self.train_labels[nearest]
    

def calculateBestFeatures(train_features, train_labels):
    featuresUsed = np.zeros(train_features.shape[1])
    for i in range(10):
        for j in range(i+1, 10):
            adata = train_features[train_labels == i, :]
            bdata = train_features[train_labels == j, :]
            dij = divergence(adata, bdata)
            sorted_indexes = np.argsort(-dij)
            features = sorted_indexes[0:5]
            for featureIndex in features:
                featuresUsed[featureIndex] += 1

    # print(featuresUsed)
    bestTenFeatures = []
    for i in range(10):
        maxIndex = np.argmax(featuresUsed)
        bestTenFeatures.append(maxIndex)
        featuresUsed[maxIndex] = -1
    return bestTenFeatures

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    returns: d12 - a vector of 1-D divergence scores
    """
    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    v1 = np.maximum(v1, 0.00000000000001)
    v2 = np.maximum(v2, 0.00000000000001)
            

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    # divergences without the need for a loop)
    # if np.all(v2 == 0): v2[:] = 0.000000000000000000000000000000001
    # if np.all(v1 == 0): v1[:] = 0.000000000000000000000000000000001

    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
    1.0 / v1 + 1.0 / v2
    )        
    return d12
        
