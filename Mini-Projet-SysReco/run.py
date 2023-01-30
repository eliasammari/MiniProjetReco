
from utility.cross_validation import split_5_folds
from configx.configx import ConfigX
from model.pmf import FunkSVDwithR
from model.social_rste import RSTE
from model.integ_svd import IntegSVD
from model.social_cune import CUNE
from model.social_reg_BENBASSOU_AMMARI import SocialReg


if __name__ == '__main__':

    rmses = []
    maes = []
    rste = SocialReg()
    for i in range(rste.config.k_fold_num):
        rste.train_model(i)
        rmse, mae = rste.predict_model()
        print("current best rmse is %0.5f, mae is %0.5f" % (rmse, mae))
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
