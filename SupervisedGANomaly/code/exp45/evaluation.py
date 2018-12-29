import numpy as np

def get_thresh(tr_output, tr_label, thresh_type):
    # tr_output为训练集样本预测分数，如[0.98, 0.78, 0.83, 0.99, 0.86, 0.23, 0.31, 0.44]
    # tr_label为训练集样本标签，OK为1，NG为0，如[1, 1, 1, 1, 1, 0, 0, 0]，ok在前，ng在后
    # thresh_type为阈值类型，共四种：'max_ok', 'max_ng', 'min_ok', 'min_ng'
    if not (thresh_type == 'max_ok' or thresh_type == 'max_ng' or thresh_type == 'min_ok' or thresh_type == 'min_ng'):
        raise TypeError

    tr_ok_num = np.sum(tr_label[:] == 1)
    tr_ng_num = np.sum(tr_label[:] == 0)
    assert(tr_ok_num + tr_ng_num == np.size(tr_label))

    if thresh_type == 'max_ok':
        thresh = np.max(tr_output[0:tr_ok_num])
    elif thresh_type == 'max_ng':
        thresh = np.max(tr_output[tr_ok_num:])
    elif thresh_type == 'min_ok':
        thresh = np.min(tr_output[0:tr_ok_num])
    elif thresh_type == 'min_ng':
        thresh = np.min(tr_output[tr_ok_num:])

    ind = np.where(tr_output == thresh)[0][0]

    return thresh, ind

def get_recall_precision(output, label, thresh_type, thresh): #训练集和测试集都可以评估
    # 这里计算tp、fp等指标和传统方法不一样，这里在乎的是负例（ng样本），而不是正例（ok样本）
    # output为训练集样本预测分数，如[0.98, 0.78, 0.83, 0.99, 0.86, 0.23, 0.31, 0.44]
    # label为训练集样本标签，OK为1，NG为0，如[1, 1, 1, 1, 1, 0, 0, 0]，ok在前，ng在后
    # thresh_type为阈值类型，共四种：'max_ok', 'max_ng', 'min_ok', 'min_ng'
    # thresh 为get_thresh函数得到的阈值
    if not (thresh_type == 'max_ok' or thresh_type == 'max_ng' or thresh_type == 'min_ok' or thresh_type == 'min_ng'):
        raise TypeError
    
    ok_num = np.sum(label[:] == 1)
    ng_num = np.sum(label[:] == 0)
    assert(ok_num + ng_num == np.size(label))

    if thresh_type == 'max_ok' or thresh_type == 'min_ng':
        count_tp = np.sum(output[ok_num:] > thresh) # for ng
        count_fn = np.sum(output[ok_num:] < thresh) # for ng
        count_fp = np.sum(output[:ok_num] > thresh) # for ok
    elif thresh_type == 'max_ng' or thresh_type =='min_ok':
        count_tp = np.sum(output[ok_num:] < thresh)
        count_fn = np.sum(output[ok_num:] > thresh)
        count_fp = np.sum(output[:ok_num] < thresh)
    
    recall = count_tp / (count_tp + count_fn)
    precision =  count_tp / (count_tp + count_fp)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1_score