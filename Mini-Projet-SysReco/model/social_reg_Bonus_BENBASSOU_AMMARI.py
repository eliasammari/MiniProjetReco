####################################  Bonus
# Ce fichier implemente l'approche 1 ( basé sur la moyenne ) qui a été detaillé dans le papier fourni


import sys

sys.path.append("..")
import numpy as np
np.seterr(invalid='ignore')

from model.mf import MF
from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp, cosine_sp, JaccSim_HD, Adar, generate_Emb, Node2Vec


class Social_Reg_HD(MF):
    """
    docstring for SocialReg with Average-based Regularization

    Ma H, Zhou D, Liu C, et al. Recommender systems with social regularization[C]//Proceedings of the fourth ACM
    international conference on Web search and data mining. ACM, 2011: 287-296.
    """

    def __init__(self):
        super(Social_Reg_HD, self).__init__()
        # self.config.lambdaP = 0.001
        # self.config.lambdaQ = 0.001
        self.config.alpha = 0.01  ################################" chenged based on the paper, old is 0.1
        self.tg = TrustGetter()
        # self.init_model()

    def init_model(self, k):
        super(Social_Reg_HD, self).init_model(k)
        self.user_sim = SimMatrix()
        print('constructing user-user similarity matrix...')
        print('Generate Embeddings (Run only once)')
        generate_Emb()

        # self.user_sim = util.load_data('../data/sim/ft_cf_soreg08_cv1.pkl')

        for u in self.rg.user:
            for f in self.tg.get_followees(u):
                if self.user_sim.contains(u, f):
                    continue
                sim = self.get_sim(u, f)
                self.user_sim.set(u, f, sim)

        # util.save_data(self.user_sim,'../data/sim/ft_cf_soreg08.pkl')

    def get_sim(self, u, k):
        sim = (cosine_sp(self.rg.get_row(u), self.rg.get_row(k)) + 1.0) / 2.0  # fit the value into range [0.0,1.0]
        return sim

    def train_model(self, k):
        super(Social_Reg_HD, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                # print('---------------------  HADDOU ---------------------')
                u = self.rg.user[user]
                # print('u', u)
                i = self.rg.item[item]
                # print('i', i)
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]
                # print( 'p, q : ', p, q)

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                pos_term, psum_sim = np.zeros((self.config.factor)), 0.0  # added
                followees = self.tg.get_followees(user)
                # print(user,'followers',followees)
                # print(followees)
                for followee in followees:
                    if self.rg.containsUser(followee):
                        s = self.user_sim[user][followee]
                        uf = self.P[self.rg.user[followee]]
                        social_term_p += s * uf
                        psum_sim += s
                #pos_term += p - np.divide(social_term_p, psum_sim)
                        pos_term += (p - (social_term_p / psum_sim))
                        social_term_loss += pos_term.dot(pos_term)  # eq 10 to chaaaaaaaaaaaage this

                social_term_m = np.zeros((self.config.factor))
                posneg_term, nsum_sim = np.zeros((self.config.factor)), 0.0
                followers = self.tg.get_followers(user)
                for follower in followers:
                    if self.rg.containsUser(follower):
                        s = self.user_sim[user][follower]
                        ug = self.P[self.rg.user[follower]]

                        followees_g = self.tg.get_followees(follower)
                        social_term_pn = np.zeros(self.config.factor)
                        for followee_g in followees_g:
                            if self.rg.containsUser(followee_g):
                                ss = self.user_sim[follower][followee_g]
                                ufn = self.P[self.rg.user[followee_g]]
                                social_term_pn += ss * ufn
                                nsum_sim += ss
                        posneg_term = social_term_pn / nsum_sim
                        social_term_m += (-s) * (ug - posneg_term) / nsum_sim

                # update latent vectors
                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * p -
                                               self.config.alpha * pos_term -
                                               self.config.alpha * social_term_m)

                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + \
                         0.5 * self.config.lambdaQ * (self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    rmses = []
    maes = []
    tcsr = Social_Reg_HD()
    # print(bmf.rg.trainSet_u[1])
    for i in range(tcsr.config.k_fold_num):
        print('the %dth cross validation training' % i)
        tcsr.train_model(i)
        rmse, mae = tcsr.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
