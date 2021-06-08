import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from empyrical.stats import max_drawdown, downside_risk, calmar_ratio

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    # print('gt_rt',np.max(ground_truth))
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long5 = 1.0
    sharpe_li5 = []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
    
        
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])


        pre_top5 = set()

        
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
        performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))

        
        # back testing on top 5
        real_ret_rat_top5 = 0
        
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)

        real_ret_rat_top5_gt = 0
        for pre in gt_top5:
            real_ret_rat_top5_gt += ground_truth[pre][i]
        real_ret_rat_top5_gt /= 5

    performance['btl5'] = bt_long5 - 1
    sharpe_li = np.array(sharpe_li)
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize
    return performance
