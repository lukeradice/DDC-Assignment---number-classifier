from utils import *
import numpy as np
from scipy.stats import multivariate_normal
import scipy.linalg


def image_to_reduced_feature(images,  labels=None, split='train'):
        # best_features_images = images[:, [np.int64(353), np.int64(69), np.int64(329), np.int64(379), np.int64(354), np.int64(318), np.int64(343), np.int64(70), np.int64(380), np.int64(71), np.int64(355), np.int64(330), np.int64(342), np.int64(317), np.int64(575), np.int64(576), np.int64(574), np.int64(44), np.int64(72), np.int64(68), np.int64(577), np.int64(67), np.int64(547), np.int64(45), np.int64(64), np.int64(599), np.int64(597), np.int64(573), np.int64(600), np.int64(392), np.int64(596), np.int64(66), np.int64(366), np.int64(598), np.int64(618), np.int64(341), np.int64(617), np.int64(578), np.int64(22), np.int64(43), np.int64(316), np.int64(73), np.int64(601), np.int64(65), np.int64(619), np.int64(164), np.int64(292), np.int64(603), np.int64(23), np.int64(46), np.int64(620), np.int64(63), np.int64(572), np.int64(546), np.int64(616), np.int64(602), np.int64(595), np.int64(579), np.int64(155), np.int64(604), np.int64(26), np.int64(62), np.int64(47), np.int64(216), np.int64(242), np.int64(621), np.int64(24), np.int64(137), np.int64(190), np.int64(622), np.int64(21), np.int64(614), np.int64(182), np.int64(25), np.int64(494), np.int64(495), np.int64(74), np.int64(27), np.int64(615), np.int64(48), np.int64(365), np.int64(163), np.int64(580), np.int64(418), np.int64(207), np.int64(41), np.int64(571), np.int64(42), np.int64(188), np.int64(605), np.int64(333), np.int64(358), np.int64(28), np.int64(340), np.int64(233), np.int64(383), np.int64(181), np.int64(309), np.int64(259), np.int64(390), np.int64(284), np.int64(49), np.int64(84), np.int64(594), np.int64(409), np.int64(285), np.int64(110), np.int64(545), np.int64(315), np.int64(50), np.int64(85), np.int64(581), np.int64(161), np.int64(520), np.int64(128), np.int64(437), np.int64(623), np.int64(208), np.int64(136), np.int64(20), np.int64(334), np.int64(189), np.int64(359), np.int64(75), np.int64(384), np.int64(624), np.int64(310), np.int64(61), np.int64(463), np.int64(187), np.int64(291), np.int64(234), np.int64(410), np.int64(606), np.int64(613), np.int64(213), np.int64(215), np.int64(260), np.int64(162), np.int64(101)]]
        print(images.shape)
        covx = np.cov(images, rowvar=0)
        print(covx.shape)
        N = covx.shape[0]
        print(N)
        w, v = scipy.linalg.eigh(covx, subset_by_index=(N - 130, N - 1))
        v = np.fliplr(v)
        pcatrain_data = np.dot((images - np.mean(images)), v)
        #top 140 - 86.6, 64.9
        # print(calculateBestFeatures(pcatrain_data, labels))
        return pcatrain_data
    # return images[:, [np.int64(353), np.int64(69), np.int64(329), np.int64(379), np.int64(354), np.int64(318), np.int64(343), np.int64(70), np.int64(380), np.int64(71), np.int64(355), np.int64(330), np.int64(342), np.int64(317), np.int64(575), np.int64(576), np.int64(574), np.int64(44), np.int64(72), np.int64(68), np.int64(577), np.int64(67), np.int64(547), np.int64(45), np.int64(64), np.int64(599), np.int64(597), np.int64(573), np.int64(600), np.int64(392), np.int64(596), np.int64(66), np.int64(366), np.int64(598), np.int64(618), np.int64(341), np.int64(617), np.int64(578), np.int64(22), np.int64(43), np.int64(316), np.int64(73), np.int64(601), np.int64(65), np.int64(619), np.int64(164), np.int64(292), np.int64(603), np.int64(23), np.int64(46), np.int64(620), np.int64(63), np.int64(572), np.int64(546), np.int64(616), np.int64(602), np.int64(595), np.int64(579), np.int64(155), np.int64(604), np.int64(26), np.int64(62), np.int64(47), np.int64(216), np.int64(242), np.int64(621), np.int64(24), np.int64(137), np.int64(190), np.int64(622), np.int64(21), np.int64(614), np.int64(182), np.int64(25), np.int64(494), np.int64(495), np.int64(74), np.int64(27), np.int64(615), np.int64(48), np.int64(365), np.int64(163), np.int64(580), np.int64(418), np.int64(207), np.int64(41), np.int64(571), np.int64(42), np.int64(188), np.int64(605), np.int64(333), np.int64(358), np.int64(28), np.int64(340), np.int64(233), np.int64(383), np.int64(181), np.int64(309), np.int64(259), np.int64(390), np.int64(284), np.int64(49), np.int64(84), np.int64(594), np.int64(409), np.int64(285), np.int64(110), np.int64(545), np.int64(315), np.int64(50), np.int64(85), np.int64(581), np.int64(161), np.int64(520), np.int64(128), np.int64(437), np.int64(623), np.int64(208), np.int64(136), np.int64(20), np.int64(334), np.int64(189), np.int64(359), np.int64(75), np.int64(384), np.int64(624), np.int64(310), np.int64(61), np.int64(463), np.int64(187), np.int64(291), np.int64(234), np.int64(410), np.int64(606), np.int64(613), np.int64(213), np.int64(215), np.int64(260), np.int64(162), np.int64(101)]]

def training_model(train_feature_vectors, train_labels):
    return ImprovedModel(train_feature_vectors, train_labels)

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_feature_vectors, train_labels):
        self.train_feature_vectors = train_feature_vectors
        self.train_labels = train_labels
        # print(calculateBestFeatures(train_feature_vectors, train_labels))

    def predict(self, test):
        
        num_0_train = self.train_feature_vectors[self.train_labels == 0]
        num_1_train = self.train_feature_vectors[self.train_labels == 1]
        num_2_train = self.train_feature_vectors[self.train_labels == 2]
        num_3_train = self.train_feature_vectors[self.train_labels == 3]
        num_4_train = self.train_feature_vectors[self.train_labels == 4]
        num_5_train = self.train_feature_vectors[self.train_labels == 5]
        num_6_train = self.train_feature_vectors[self.train_labels == 6]
        num_7_train = self.train_feature_vectors[self.train_labels == 7]
        num_8_train = self.train_feature_vectors[self.train_labels == 8]
        num_9_train = self.train_feature_vectors[self.train_labels == 9]
        
        mean_0 = np.mean(num_0_train, axis=0)
        mean_1 = np.mean(num_1_train, axis=0)
        mean_2 = np.mean(num_2_train, axis=0)
        mean_3 = np.mean(num_3_train, axis=0)
        mean_4 = np.mean(num_4_train, axis=0)
        mean_5 = np.mean(num_5_train, axis=0)
        mean_6 = np.mean(num_6_train, axis=0)
        mean_7 = np.mean(num_7_train, axis=0)
        mean_8 = np.mean(num_8_train, axis=0)
        mean_9 = np.mean(num_9_train, axis=0)

        reg_strength = 1185

        cov_0 = np.cov(num_0_train, rowvar = 0)
        pd_cov_0 = cov_0 +  np.eye(cov_0.shape[0]) * reg_strength
        cov_1 = np.cov(num_1_train, rowvar = 0)
        pd_cov_1 = cov_1 +  np.eye(cov_1.shape[0]) * reg_strength
        cov_2 = np.cov(num_2_train, rowvar = 0)
        pd_cov_2 = cov_2 +  np.eye(cov_2.shape[0]) * reg_strength
        cov_3 = np.cov(num_3_train, rowvar = 0)
        pd_cov_3 = cov_3 +  np.eye(cov_3.shape[0]) * reg_strength
        cov_4 = np.cov(num_4_train, rowvar = 0)
        pd_cov_4 = cov_4 +  np.eye(cov_4.shape[0]) * reg_strength
        cov_5 = np.cov(num_5_train, rowvar = 0)
        pd_cov_5 = cov_5 +  np.eye(cov_5.shape[0]) * reg_strength
        cov_6 = np.cov(num_6_train, rowvar = 0)
        pd_cov_6 = cov_6 +  np.eye(cov_6.shape[0]) * reg_strength
        cov_7 = np.cov(num_7_train, rowvar = 0)
        pd_cov_7 = cov_7 +  np.eye(cov_7.shape[0]) * reg_strength
        cov_8 = np.cov(num_8_train, rowvar = 0)
        pd_cov_8 = cov_8 +  np.eye(cov_8.shape[0]) * reg_strength
        cov_9 = np.cov(num_9_train, rowvar = 0)
        pd_cov_9 = cov_9 +  np.eye(cov_9.shape[0]) * reg_strength
        

        dist0 = multivariate_normal(mean=mean_0, cov=pd_cov_0)
        dist1 = multivariate_normal(mean=mean_1, cov=pd_cov_1)
        dist2 = multivariate_normal(mean=mean_2, cov=pd_cov_2)
        dist3 = multivariate_normal(mean=mean_3, cov=pd_cov_3)
        dist4 = multivariate_normal(mean=mean_4, cov=pd_cov_4)
        dist5 = multivariate_normal(mean=mean_5, cov=pd_cov_5)
        dist6 = multivariate_normal(mean=mean_6, cov=pd_cov_6)
        dist7 = multivariate_normal(mean=mean_7, cov=pd_cov_7)
        dist8 = multivariate_normal(mean=mean_8, cov=pd_cov_8)
        dist9 = multivariate_normal(mean=mean_9, cov=pd_cov_9)

        p0 = (dist0.pdf(test))
        p1 = (dist1.pdf(test))
        p2 = (dist2.pdf(test))
        p3 = (dist3.pdf(test))
        p4 = (dist4.pdf(test))
        p5 = (dist5.pdf(test))
        p6 = (dist6.pdf(test))
        p7 = (dist7.pdf(test))
        p8 = (dist8.pdf(test))
        p9 = (dist9.pdf(test))
        probability = np.vstack((p0, p1, p2, p3, p4, p5, p6, p7, p8, p9))
        return np.argmax(probability, axis=0)
    
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
    for i in range(138):
        maxIndex = np.argmax(scores)
        bestFeatures.append(maxIndex)
        scores[maxIndex] = -1
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
        
