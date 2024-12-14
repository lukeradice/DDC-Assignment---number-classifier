from utils import *
import numpy as np
from scipy.stats import multivariate_normal, mode
import scipy.linalg

featuresSel = [np.int64(353), np.int64(69), np.int64(329), np.int64(70), np.int64(354), np.int64(379), np.int64(318), np.int64(343), np.int64(71), np.int64(575), np.int64(576), np.int64(380), np.int64(355), np.int64(330), np.int64(574), np.int64(342), np.int64(317), np.int64(577), np.int64(44), np.int64(68), np.int64(72), np.int64(600), np.int64(599), np.int64(67), np.int64(597), np.int64(618), np.int64(45), np.int64(598), np.int64(573), np.int64(596), np.int64(617), np.int64(547), np.int64(578), np.int64(64), np.int64(601), np.int64(66), np.int64(619), np.int64(620), np.int64(603), np.int64(602), 
np.int64(392), np.int64(366), np.int64(22), np.int64(43), np.int64(73), np.int64(341), np.int64(621), np.int64(23), np.int64(616), np.int64(164), np.int64(65), np.int64(46), np.int64(155), np.int64(579), np.int64(316), np.int64(604), np.int64(595), np.int64(572), np.int64(622), np.int64(182), np.int64(292), np.int64(26), np.int64(63), np.int64(47), np.int64(546), np.int64(24), np.int64(614), np.int64(27), np.int64(216), np.int64(25), np.int64(615), np.int64(62), np.int64(21), np.int64(190), np.int64(137), np.int64(242), np.int64(74), np.int64(48), np.int64(580), np.int64(605), np.int64(188), np.int64(163), np.int64(571), np.int64(181), np.int64(41), np.int64(28), np.int64(42), np.int64(207), np.int64(495), np.int64(494), np.int64(365), np.int64(623), np.int64(418), np.int64(594), np.int64(128), np.int64(49), np.int64(233), np.int64(581), np.int64(333), np.int64(84), np.int64(340), np.int64(358), np.int64(208), np.int64(161), np.int64(383), np.int64(624), np.int64(110), np.int64(545), np.int64(20), np.int64(50), np.int64(638), np.int64(390), np.int64(75), np.int64(259), np.int64(309), np.int64(85), np.int64(187), np.int64(315), np.int64(284), np.int64(136), np.int64(637), np.int64(189), np.int64(606), np.int64(285), np.int64(409), np.int64(520), np.int64(613), np.int64(209), np.int64(437), np.int64(101), np.int64(213), np.int64(234), np.int64(61), np.int64(40), np.int64(310), np.int64(334), np.int64(215), np.int64(291), np.int64(570), np.int64(162), np.int64(359), np.int64(463), np.int64(384), np.int64(214), np.int64(410), np.int64(625), np.int64(582), np.int64(39), np.int64(241), np.int64(260), np.int64(266), np.int64(607), np.int64(290), np.int64(127), np.int64(51), np.int64(635), np.int64(100), np.int64(411), np.int64(314), np.int64(19), np.int64(493), np.int64(626), np.int64(109), np.int64(488), np.int64(60), np.int64(385), np.int64(519), np.int64(77), np.int64(636), np.int64(389), np.int64(83), np.int64(30), np.int64(134), np.int64(135), np.int64(514), np.int64(639), np.int64(443), np.int64(538), np.int64(240), np.int64(76), np.int64(102), np.int64(235), np.int64(544), np.int64(468), np.int64(29), np.int64(265), np.int64(489), np.int64(360), np.int64(439), np.int64(335), np.int64(543), np.int64(593), np.int64(52), np.int64(186), np.int64(160), np.int64(438), np.int64(612), np.int64(31), np.int64(364), np.int64(583), np.int64(59), np.int64(640), np.int64(53), np.int64(156), np.int64(569), np.int64(183), np.int64(537), np.int64(107), np.int64(129), np.int64(513), np.int64(608), np.int64(82), np.int64(38), np.int64(239), np.int64(363), np.int64(417), np.int64(339), np.int64(641), np.int64(464), np.int64(212), np.int64(6), np.int64(108), np.int64(210), np.int64(11), np.int64(633), np.int64(58), np.int64(632), np.int64(54), np.int64(37), np.int64(518), np.int64(562), np.int64(634), np.int64(338), np.int64(539), np.int64(3), np.int64(81), np.int64(490), np.int64(32), np.int64(631), np.int64(18), np.int64(592), np.int64(542), np.int64(492), np.int64(312), np.int64(627), np.int64(561), np.int64(133), np.int64(643), np.int64(9), np.int64(311), np.int64(264), np.int64(386), np.int64(568), np.int64(412), np.int64(4), np.int64(465), np.int64(78), np.int64(630), np.int64(584), np.int64(585), np.int64(313), np.int64(567), np.int64(563), np.int64(16), np.int64(261), np.int64(289), np.int64(8), np.int64(517), np.int64(2), np.int64(185), np.int64(515), np.int64(33), np.int64(236), np.int64(416), np.int64(586), np.int64(159), np.int64(103), np.int64(17), np.int64(467), np.int64(336), np.int64(611), np.int64(591), np.int64(36), np.int64(132), np.int64(286), np.int64(263), np.int64(262), np.int64(388), np.int64(157), np.int64(13), np.int64(541), np.int64(361), np.int64(566), np.int64(57), np.int64(12), np.int64(5), np.int64(238), np.int64(7), np.int64(79), np.int64(440), np.int64(442), np.int64(80), np.int64(106), np.int64(131), np.int64(10), np.int64(211), np.int64(540), np.int64(14), np.int64(35), np.int64(55), np.int64(642), np.int64(34), np.int64(588), np.int64(184), np.int64(413), np.int64(590), np.int64(469), np.int64(362), np.int64(564), np.int64(15), np.int64(609), np.int64(644), np.int64(287), np.int64(512), np.int64(288), np.int64(154), np.int64(587), np.int64(441), np.int64(521), np.int64(414), np.int64(629), np.int64(0), np.int64(393), np.int64(337), np.int64(56), np.int64(610), np.int64(237), np.int64(130), np.int64(565), np.int64(368), np.int64(158), np.int64(415), np.int64(344), np.int64(105), np.int64(589), np.int64(293), np.int64(1), np.int64(516), np.int64(192), np.int64(444), np.int64(180), np.int64(367), np.int64(126), np.int64(267), np.int64(191), np.int64(165), np.int64(387), np.int64(319), np.int64(378), np.int64(328), np.int64(466), np.int64(470), np.int64(257), np.int64(356), np.int64(381), np.int64(104), np.int64(231), np.int64(407), np.int64(331), np.int64(405), np.int64(406), np.int64(491), np.int64(206), np.int64(628), np.int64(391), np.int64(446), np.int64(232), np.int64(357), np.int64(382), np.int64(258), np.int64(332), np.int64(96), np.int64(268), np.int64(408), np.int64(99), np.int64(419), np.int64(283), np.int64(352), np.int64(436), np.int64(294), np.int64(548), np.int64(560), np.int64(166), np.int64(404), np.int64(308), np.int64(377), np.int64(256), np.int64(91), np.int64(179), np.int64(445), np.int64(90), np.int64(496), np.int64(86), np.int64(217), np.int64(282), np.int64(94), np.int64(193), np.int64(471), np.int64(306), np.int64(497), np.int64(138), np.int64(536), np.int64(420), np.int64(243), np.int64(295), np.int64(111), np.int64(93), np.int64(325), np.int64(522), np.int64(487), np.int64(374), np.int64(349), np.int64(345), np.int64(373), np.int64(92), np.int64(97), np.int64(305), np.int64(307), np.int64(350), np.int64(369), np.int64(324), np.int64(394), np.int64(270), np.int64(472), np.int64(269), np.int64(95), np.int64(281), np.int64(433), np.int64(301), np.int64(421), np.int64(246), np.int64(473), np.int64(348), np.int64(435), np.int64(323), np.int64(399), np.int64(447), np.int64(300), np.int64(280), np.int64(474), np.int64(205), np.int64(434), np.int64(245), np.int64(302), np.int64(320), np.int64(153), np.int64(372), np.int64(549), np.int64(462), np.int64(326), np.int64(218), np.int64(125), np.int64(304), np.int64(398), np.int64(219), np.int64(98), np.int64(327), np.int64(375), np.int64(498), np.int64(178), np.int64(220), np.int64(87), np.int64(244), np.int64(432), np.int64(551), np.int64(400), np.int64(321), np.int64(550), np.int64(351), np.int64(271), np.int64(167), np.int64(322), np.int64(276), np.int64(448), np.int64(559), np.int64(177), np.int64(299), np.int64(89), np.int64(112), np.int64(275), np.int64(194), np.int64(113), np.int64(247), np.int64(296), np.int64(422), np.int64(449), np.int64(459), np.int64(347), np.int64(523), np.int64(557), np.int64(221), np.int64(139), np.int64(558), np.int64(458), np.int64(277), np.int64(499), np.int64(176), np.int64(552), np.int64(460), np.int64(461), np.int64(140), np.int64(511), np.int64(450), np.int64(303), np.int64(297), np.int64(195), np.int64(555), np.int64(88), np.int64(556), np.int64(425), np.int64(486), np.int64(152), np.int64(475), np.int64(272), np.int64(395), np.int64(426), np.int64(230), np.int64(427), np.int64(251), np.int64(535), np.int64(202), np.int64(204), np.int64(424), np.int64(298), np.int64(451), np.int64(376), np.int64(250), np.int64(553), np.int64(510), np.int64(203), np.int64(485), np.int64(403), np.int64(255), np.int64(201), np.int64(397), np.int64(124), np.int64(554), np.int64(370), np.int64(431), np.int64(401), np.int64(279), np.int64(484), np.int64(534), np.int64(278), np.int64(524), np.int64(346), np.int64(423), np.int64(151), np.int64(428), np.int64(452), np.int64(175), np.int64(225), np.int64(227), np.int64(252), np.int64(118), np.int64(168), np.int64(274), np.int64(483), np.int64(396), np.int64(226), np.int64(500), np.int64(224), 
np.int64(457), np.int64(476), np.int64(117), np.int64(371), np.int64(119), np.int64(150), np.int64(509), np.int64(453), np.int64(273), np.int64(149), np.int64(228), np.int64(199), np.int64(525), np.int64(200), np.int64(248), np.int64(123), np.int64(114), np.int64(477), np.int64(253), np.int64(229), np.int64(254), np.int64(249), np.int64(508), np.int64(196), np.int64(402), np.int64(533), np.int64(222), np.int64(120), np.int64(145), np.int64(526), np.int64(172), np.int64(198), np.int64(478), np.int64(116), np.int64(141), np.int64(174), np.int64(148), np.int64(122), np.int64(429), np.int64(527), np.int64(121), np.int64(173), np.int64(482), np.int64(115), np.int64(171), np.int64(501), np.int64(144), np.int64(146), np.int64(454), np.int64(528), np.int64(197), np.int64(529), np.int64(147), np.int64(479), np.int64(169), np.int64(507), np.int64(530), np.int64(170), np.int64(223), np.int64(532), np.int64(502), np.int64(430), np.int64(531), np.int64(481), np.int64(480), np.int64(503), np.int64(143), np.int64(506), np.int64(504), np.int64(455), np.int64(142), np.int64(456), np.int64(505)]

def image_to_reduced_feature(images,  labels=None, split='train'):
    # mean = np.mean(images, axis=0)
    # std = np.std(images, axis=0)
    # standardised_images = (images - mean)/std
    # #top 140 - 86.6, 64.9
    # return standardised_images[:, :49]
 
        #  test = test / np.linalg.norm(test, axis=1, keepdims=True)
        # train = train / np.linalg.norm(train, axis=1, keepdims=True)
    normalised =  images / np.linalg.norm(images, axis=1, keepdims=True)
    # return normalised[:, featuresSel]
    return normalised
    # return images[:, [np.int64(353), np.int64(69), np.int64(329), np.int64(379), np.int64(354), np.int64(318), np.int64(343), np.int64(70), np.int64(380), np.int64(71), np.int64(355), np.int64(330), np.int64(342), np.int64(317), np.int64(575), np.int64(576), np.int64(574), np.int64(44), np.int64(72), np.int64(68), np.int64(577), np.int64(67), np.int64(547), np.int64(45), np.int64(64), np.int64(599), np.int64(597), np.int64(573), np.int64(600), np.int64(392), np.int64(596), np.int64(66), np.int64(366), np.int64(598), np.int64(618), np.int64(341), np.int64(617), np.int64(578), np.int64(22), np.int64(43), np.int64(316), np.int64(73), np.int64(601), np.int64(65), np.int64(619), np.int64(164), np.int64(292), np.int64(603), np.int64(23), np.int64(46), np.int64(620), np.int64(63), np.int64(572), np.int64(546), np.int64(616), np.int64(602), np.int64(595), np.int64(579), np.int64(155), np.int64(604), np.int64(26), np.int64(62), np.int64(47), np.int64(216), np.int64(242), np.int64(621), np.int64(24), np.int64(137), np.int64(190), np.int64(622), np.int64(21), np.int64(614), np.int64(182), np.int64(25), np.int64(494), np.int64(495), np.int64(74), np.int64(27), np.int64(615), np.int64(48), np.int64(365), np.int64(163), np.int64(580), np.int64(418), np.int64(207), np.int64(41), np.int64(571), np.int64(42), np.int64(188), np.int64(605), np.int64(333), np.int64(358), np.int64(28), np.int64(340), np.int64(233), np.int64(383), np.int64(181), np.int64(309), np.int64(259), np.int64(390), np.int64(284), np.int64(49), np.int64(84), np.int64(594), np.int64(409), np.int64(285), np.int64(110), np.int64(545), np.int64(315), np.int64(50), np.int64(85), np.int64(581), np.int64(161), np.int64(520), np.int64(128), np.int64(437), np.int64(623), np.int64(208), np.int64(136), np.int64(20), np.int64(334), np.int64(189), np.int64(359), np.int64(75), np.int64(384), np.int64(624), np.int64(310), np.int64(61), np.int64(463), np.int64(187), np.int64(291), np.int64(234), np.int64(410), np.int64(606), np.int64(613), np.int64(213), np.int64(215), np.int64(260), np.int64(162), np.int64(101)]]
    # return standardised_images

def training_model(train_feature_vectors, train_labels):
    return ImprovedModel(train_feature_vectors, train_labels)

class ImprovedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, train_feature_vectors, train_labels):
        self.train_feature_vectors = train_feature_vectors
        self.train_labels = train_labels
        # print(calculateBestFeatures(train_feature_vectors, train_labels))
    
    def project_pca_train(self, test_data):
        covx = np.cov(self.train_feature_vectors, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, subset_by_index=(N - 240, N - 1))
        v = np.fliplr(v)
        return np.dot((test_data - np.mean(self.train_feature_vectors, axis=0)), v)
        return test_data
    #without pca = 93.2, 79.8
    #250 = 93.50, 77.9 with k=1
    #200: 93.1, 77.4
    #250: 93.3, 77.4 with k=3
 
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
        train = self.project_pca_train(self.train_feature_vectors) 
        test = self.project_pca_train(test)

        # Super compact implementation of nearest neighbour
        x = np.dot(test, train.transpose())
        modtest = np.sqrt(np.sum(test * test, axis=1))
        modtrain = np.sqrt(np.sum(train * train, axis=1))
        # cosine distance
        dist = x / np.outer(modtest, modtrain.transpose())
        return self.getKNearest(3, dist)

    def predict2(self, test):
        
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

        # reg_strength = 1675
        # reg_strength = 1185
        reg_strength = 0

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
    for i in range(645):
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
        
