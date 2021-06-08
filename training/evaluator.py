import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    print('gt_rt',np.max(ground_truth))
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    bt_long2 = 1.0
    bt_long7 = 1.0

    top_1_ground_truth = []
    top_5_ground_truth = []
    top_10_ground_truth = []
    sharpe_li = []
    sharpe_li2 = []
    sharpe_li5 = []
    sharpe_li7 = []
    sharpe_li10= []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        gt_top2 = set()
        gt_top7 = set()
        
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
            if len(gt_top2) < 2:
                gt_top2.add(cur_rank)
            if len(gt_top7) < 7:
                gt_top7.add(cur_rank)
            

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        pre_top2 = set()
        pre_top7 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
            if len(pre_top2) < 2:
                pre_top2.add(cur_rank)
            if len(pre_top7) < 7:
                pre_top7.add(cur_rank)
            
        performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))
        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # back testing on top 1
        aaabbb = ground_truth[list(gt_top1)[0]][i]
        top_1_ground_truth.append(aaabbb)

        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        sharpe_li.append(real_ret_rat_top)
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
        top_5_ground_truth.append(real_ret_rat_top5_gt)
        # back testing on top 10

        # back testing on top 2
        real_ret_rat_top2 = 0
        for pre in pre_top2:
            real_ret_rat_top2 += ground_truth[pre][i]
        real_ret_rat_top2 /= 2
        bt_long2 += real_ret_rat_top2
        sharpe_li2.append(real_ret_rat_top2)

        # back testing on top 7
        real_ret_rat_top7 = 0
        for pre in pre_top7:
            real_ret_rat_top7 += ground_truth[pre][i]
        real_ret_rat_top7 /= 7
        bt_long7 += real_ret_rat_top7
        sharpe_li7.append(real_ret_rat_top7)



        real_ret_rat_top10_gt = 0
        real_ret_rat_top10 = 0
        for pre in gt_top10:
            real_ret_rat_top10_gt += ground_truth[pre][i]
        real_ret_rat_top10_gt /= 10
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
        sharpe_li10.append(real_ret_rat_top10)
        top_10_ground_truth.append(real_ret_rat_top10_gt)

    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    performance['btl5'] = bt_long5
    performance['btl10'] = bt_long10
    sharpe_li = np.array(sharpe_li)
    performance['sharpe1'] = np.mean(sharpe_li)/np.std(sharpe_li)
    performance['sharpe2'] = np.mean(sharpe_li2)/np.std(sharpe_li2)
    performance['sharpe5'] = np.mean(sharpe_li5)/np.std(sharpe_li5)
    performance['sharpe7'] = np.mean(sharpe_li7)/np.std(sharpe_li7)
    performance['sharpe10'] = np.mean(sharpe_li10)/np.std(sharpe_li10)
    return performance
