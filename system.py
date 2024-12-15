from utils import *
import numpy as np
from scipy.stats import mode
import scipy.linalg

top_55_features_after_pca = [np.int64(0), np.int64(1), np.int64(3), np.int64(4), np.int64(6), np.int64(5), np.int64(2), np.int64(12), np.int64(15), np.int64(7), np.int64(13), np.int64(9), np.int64(16), np.int64(8), np.int64(18), np.int64(41), np.int64(11), np.int64(14), np.int64(17), np.int64(21), np.int64(20), np.int64(19), np.int64(23), np.int64(26), np.int64(22), np.int64(33), np.int64(37), np.int64(31), np.int64(53), np.int64(25), np.int64(10), np.int64(28), np.int64(32), np.int64(36), np.int64(40), np.int64(24), np.int64(82), np.int64(46), np.int64(62), np.int64(95), np.int64(38), np.int64(65), np.int64(27), np.int64(30), np.int64(64), np.int64(50), np.int64(58), np.int64(45), np.int64(168), np.int64(68), np.int64(98), np.int64(94), np.int64(60), np.int64(99), np.int64(39)]
# top_55_features_after_pca = [np.int64(0), np.int64(1), np.int64(3), np.int64(6), np.int64(5), np.int64(2), np.int64(15), np.int64(36), np.int64(13), np.int64(7), np.int64(40), np.int64(45), np.int64(126), np.int64(33), np.int64(53), np.int64(4), np.int64(23), np.int64(37), np.int64(16), np.int64(94), np.int64(105), np.int64(43), np.int64(25), np.int64(213), np.int64(109), np.int64(144), np.int64(156), np.int64(11), np.int64(28), np.int64(24), np.int64(31), np.int64(82), np.int64(99), np.int64(118), np.int64(123), np.int64(151), np.int64(181), np.int64(17), np.int64(87), np.int64(124), np.int64(168), np.int64(136), np.int64(157), np.int64(101), np.int64(180), np.int64(107), np.int64(160), np.int64(102), np.int64(179), np.int64(18), np.int64(166), np.int64(10), np.int64(129), np.int64(203), np.int64(47)]
def apply_pca(images):
    train_images, train_labels = get_dataset('train')  
    covx = np.cov(train_images, rowvar=0)
    N = covx.shape[0]
    # get the top 240 principal components
    w, v = scipy.linalg.eigh(covx, subset_by_index=(N - 240, N - 1))
    v = np.fliplr(v)
    return np.dot((images - np.mean(train_images, axis=0)), v)

def image_to_reduced_feature(images,  labels=None, split=None):
    if split=='train': 
        #if this is conversion of training data it duplicates the data but  
        #masked to addour training data to aid classifying the masked data
        mask =  np.ones((28, 28), dtype=int)
        mask[6:21, 6:21] = 0 # central black mask
        masked_image_simulation = images * mask.flatten() #applying to training
        # adding it on top of the training data we already had
        images = np.concatenate([images, masked_image_simulation], axis=0)
        #IF THE ABOVE ISN'T ALLOWED COMMENTING IT OUT IS FINE, DOESN'T AFFECT
        #ACCURACY TOO MUCH 
    # normalise the data so calculting cosine distance in knn will be effective
    normalised =  images / np.linalg.norm(images, axis=1, keepdims=True)
    pca_applied =  apply_pca(normalised)
    return pca_applied[:, top_55_features_after_pca]

def training_model(train_feature_vectors, train_labels):
    return ImprovedModel(train_feature_vectors, train_labels)

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        print(train_data.shape)
        self.train_labels = np.concatenate([train_labels, train_labels], axis=0)
        # print(calculateBestFeatures(train_data, self.train_labels))
 
    def getKNearest(self, k, dist):
        nearestTracker = []
        nearest = np.argmax(dist, axis=1)
        for i in range(k):
            nearestLabel = self.train_labels[nearest]
            nearestTracker.append(nearestLabel)
            dist[np.arange(dist.shape[0]), nearest] = -np.inf
            nearest = np.argmax(dist, axis=1)
        nearestTrackerNp = np.array(nearestTracker)
        computeMode = mode(nearestTrackerNp, axis=0)
        return computeMode.mode.flatten()
        #best = 4: 93.30, 80.80

    def predict(self, test):
        # nearest neighbour
        x = np.dot(test, self.train_data.transpose())
        modtest = np.sqrt(np.sum(test * test, axis=1))
        modtrain = np.sqrt(np.sum(self.train_data * self.train_data, axis=1))
        # cosine distance
        dist = x / np.outer(modtest, modtrain.transpose())
        return self.getKNearest(3, dist)
        
def calculateBestFeatures(train_features, train_labels):
    variances = np.var(train_features, axis=0)
    train_features = train_features[:, variances > 1e-10]
    scores = np.zeros(train_features.shape[1])
    for num1 in np.arange(10):
        num1_data = train_features[train_labels == num1, :]
        for num2 in np.arange(num1+1, 10):
            num2_data = train_features[train_labels == num2, :]
            d12 = divergence(num1_data, num2_data)
            scores += d12

    bestFeatures = []
    for i in range(55):
        maxIndex = np.argmax(scores)
        bestFeatures.append(maxIndex)
        scores[maxIndex] = -np.inf
    return bestFeatures

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

    v1 = np.maximum(v1, 1e-10)
    v2 = np.maximum(v2, 1e-10)
            

    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
    1.0 / v1 + 1.0 / v2
    )        
    return d12
        
