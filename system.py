from utils import *
import numpy as np
from scipy.stats import mode
import scipy.linalg


def apply_pca(images):
    #normalised data is projected onto non normalised test data
    train_images, train_labels = get_dataset('train') 
    covx = np.cov(train_images, rowvar=0)
    N = covx.shape[0]
    # get the top 170 principal components
    w, v = scipy.linalg.eigh(covx, subset_by_index=(N - 170, N - 1))
    v = np.fliplr(v)
    return np.dot((images - np.mean(train_images, axis=0)), v)

def image_to_reduced_feature(images,  labels=None, split=None):
    # normalise the data so calculting cosine distance in knn will be effective
    # normalised =  images / np.linalg.norm(images, axis=1, keepdims=True)
    norms = np.linalg.norm(images, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # Prevent divide-by-zero
    normalised = images / norms
    pca_applied =  apply_pca(normalised)
    return pca_applied

def training_model(train_feature_vectors, train_labels):
    return ImprovedModel(train_feature_vectors, train_labels)

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

        # below is what I used to determine best features, obviously no need for
        # it to be called in my final classifier but keeping it here to show you
 
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

    def predict(self, test):
        # nearest neighbour
        x = np.dot(test, self.train_data.transpose())
        modtest = np.sqrt(np.sum(test * test, axis=1))
        modtrain = np.sqrt(np.sum(self.train_data * self.train_data, axis=1))
        modtest[modtest == 0] = 1e-10
        modtrain[modtrain == 0] = 1e-10
        # cosine distance
        dist = x / np.outer(modtest, modtrain.transpose())
        return self.getKNearest(3, dist)