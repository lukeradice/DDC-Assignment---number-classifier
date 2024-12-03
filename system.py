from utils import *
import numpy as np
from scipy.stats import multivariate_normal


def image_to_reduced_feature(images, split='train'):
    # tenBestItOne = [129, 438, 128, 98, 411, 652, 684, 713, 127, 221]
    # tenBestItTwo = [128, 130, 438, 129, 411, 684, 127, 683, 713, 98]
    twentyBestItOne = [128, 130, 438, 129, 411, 684, 127, 683, 713, 98, 400, 439, 467, 652, 685, 712, 99, 221, 248, 249]
    # twentyBestItTwo = [np.int64(128), np.int64(130), np.int64(129), np.int64(98), np.int64(131), np.int64(69), np.int64(411), np.int64(438), np.int64(439), np.int64(467), np.int64(70), np.int64(99), np.int64(249), np.int64(683), np.int64(684), np.int64(685), np.int64(712), np.int64(126), np.int64(127), np.int64(193)]
#     thirtyBestItOne = [np.int64(128), np.int64(130), np.int64(129), np.int64(98), np.int64(131), np.int64(69), np.int64(411), np.int64(438), np.int64(439), np.int64(467), np.int64(70), np.int64(99), np.int64(249), np.int64(683), np.int64(684), np.int64(685), np.int64(712), np.int64(126), np.int64(127), np.int64(193), np.int64(221), np.int64(400), np.int64(428), 
# np.int64(592), np.int64(686), np.int64(711), np.int64(713), np.int64(427), np.int64(468), np.int64(652)]
    # thirtyBestItTwo = [np.int64(128), np.int64(130), np.int64(438), np.int64(129), np.int64(411), np.int64(684), np.int64(127), np.int64(683), np.int64(713), np.int64(98), np.int64(400), np.int64(439), np.int64(467), np.int64(652), np.int64(685), np.int64(712), np.int64(99), np.int64(221), np.int64(248), np.int64(249), np.int64(399), np.int64(428), np.int64(468), np.int64(592), np.int64(651), np.int64(714), np.int64(97), np.int64(123), np.int64(124), np.int64(125)]
    return images[:, twentyBestItOne]
    return images

def training_model(train_feature_vectors, train_labels):
    
    return ImprovedModel(train_feature_vectors, train_labels)
    # return NullModel()
    # null model takes in the training data and has a method predict
    # which returns an estimated class label for each row that is input

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_feature_vectors, train_labels):
       self.distList = self.getDistributionObjectForEachClass(
                                            train_feature_vectors, train_labels)
    #    print(calculateBestFeatures(train_feature_vectors, train_labels))

    def getDistributionObjectForEachClass(self, train_feature_vectors, train_labels):
        dists = []
        for i in range(10):
            num_x_train = train_feature_vectors[train_labels == i]
            mean_x = np.mean(num_x_train, axis=0)
            cov_x = np.cov(num_x_train, rowvar = 0)
            dists.append(multivariate_normal(mean=mean_x, cov=cov_x))


    def predict(self, test):
        probabilityForEachClass = []
        for i in range(10):
            probabilityForEachClass.append(self.distList[i].pdf(test))
        probability = np.vstack(probabilityForEachClass)
        return np.argmax(probability, axis=0)
    

# def calculateBestFeatures(train_features, train_labels):
#     featuresUsed = np.zeros(train_features.shape[1])
#     for i in range(10):
#         for j in range(i+1, 10):
#             adata = train_features[train_labels == i, :]
#             bdata = train_features[train_labels == j, :]
#             dij = divergence(adata, bdata)
#             sorted_indexes = np.argsort(-dij)
#             features = sorted_indexes[0:5]
#             for featureIndex in features:
#                 featuresUsed[featureIndex] += 1

#     # print(featuresUsed)
#     bestTenFeatures = []
#     for i in range(30):
#         maxIndex = np.argmax(featuresUsed)
#         bestTenFeatures.append(maxIndex)
#         featuresUsed[maxIndex] = -1
#     return bestTenFeatures

# def divergence(class1, class2):
#     """compute a vector of 1-D divergences
#     class1 - data matrix for class 1, each row is a sample
#     class2 - data matrix for class 2
#     returns: d12 - a vector of 1-D divergence scores
#     """
#     # Compute the mean and variance of each feature vector element
#     m1 = np.mean(class1, axis=0)
#     m2 = np.mean(class2, axis=0)
#     v1 = np.var(class1, axis=0)
#     v2 = np.var(class2, axis=0)

#     v1 = np.maximum(v1, 0.00000000000001)
#     v2 = np.maximum(v2, 0.00000000000001)
            

#     # Plug mean and variances into the formula for 1-D divergence.
#     # (Note that / and * are being used to compute multiple 1-D
#     # divergences without the need for a loop)
#     # if np.all(v2 == 0): v2[:] = 0.000000000000000000000000000000001
#     # if np.all(v1 == 0): v1[:] = 0.000000000000000000000000000000001

#     d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
#     1.0 / v1 + 1.0 / v2
#     )        
#     return d12
        
