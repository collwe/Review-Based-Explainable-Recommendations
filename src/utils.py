import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch

from model_CAR import CAR
from model_CER import CER
from model_CNR import CNR_Anchor, CNR_Intervener
from model_NAR import NAR
from src.model_NCF import NCF
from src.model_VBPR import VBPR

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shuffle(train_users, train_users_feature, train_pos_items, train_pos_items_feature, train_neg_items, train_neg_items_feature):
    train_records_num = len(train_users)
    index = np.array(range(train_records_num)).astype(int)
    np.random.shuffle(index)
    input_user = list(np.array(train_users)[index])
    input_user_feature = list(np.array(train_users_feature)[index])
    input_pos_item = list(np.array(train_pos_items)[index])
    input_pos_item_feature = list(np.array(train_pos_items_feature)[index])
    input_neg_item = list(np.array(train_neg_items)[index])
    input_neg_item_feature = list(np.array(train_neg_items_feature)[index])

    return input_user, input_user_feature, input_pos_item, input_pos_item_feature, input_neg_item, input_neg_item_feature


class Evaluate(object):
    def __init__(self, topk):
        self.Top_K = topk

    def MAP(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    hit += 1
                    tmp += hit / (j + 1)
            result.append(tmp)
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            temp = 0
            Z_u = 0

            for j in range(min(len(fit), len(v))):
                Z_u = Z_u + 1 / np.log2(j + 2)
            for j in range(len(fit)):
                if fit[j] in v:
                    temp = temp + 1 / np.log2(j + 2)

            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def top_k(self, ground_truth, pred):
        p_total = []
        r_total = []
        f_total = []
        hit_total = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            cross = float(len([i for i in fit if i in v]))
            p = cross / len(fit)
            r = cross / len(v)
            if cross > 0:
                f = 2.0 * p * r / (p + r)
            else:
                f = 0.0
            hit = 1.0 if cross > 0 else 0.0
            p_total.append(p)
            r_total.append(r)
            f_total.append(f)
            hit_total.append(hit)
        return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f_total).mean(), np.array(hit_total).mean()


    def evaluate(self, ground_truth, pred):
        sorted_pred = {}
        for k, v in pred.items():
            sorted_pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]

        p, r, f1, hit = self.top_k(ground_truth, sorted_pred)
        map = self.MAP(ground_truth, sorted_pred)
        mrr = self.MRR(ground_truth, sorted_pred)
        ndcg = self.NDCG(ground_truth, sorted_pred)
        return map, mrr, p, r, f1, hit, ndcg



def load_word_embedding(debug=False):
    if debug:
        return {}, np.zeros((10000, 300))
    else:
        lines = open(parent_path + '/data/glove.6B.300d.txt').readlines()
        data = []
        word_dict = {}
        for idx, line in enumerate(lines):
            tokens = line.strip('\n')
            # word, vec = tokens.split(' ')
            tokens = tokens.split(' ')
            word = tokens[0]
            vec_nums = tokens[1:]
            # vec_nums = vec.split(' ')
            word_dict[word] = idx
            temp_ = [float(i) for i in vec_nums]
            assert len(temp_) == 300
            data.append(temp_)
        data = np.array(data)
        assert data.shape[1] == 300
        print("Loaded data. #shape = " + str(data.shape))
        print(" #words = %d " % (len(word_dict)))
        return word_dict, data



def init_model(args, data):
    evaluator = Evaluate(args.K)
    model, i_model = None, None
    if args.model_name == 'CER':
        args.out_loop = 1
        model = CER(args, data)
    if args.model_name == 'NCF':
        args.out_loop = 1
        model = NCF(args, data)
    if args.model_name == 'VBPR':
        args.out_loop = 1
        model = VBPR(args, data)
    if args.model_name == 'CAR':
        args.out_loop = 1
        args.adv_epoch = args.epochs // 2
        model = CAR(args, data)
    if args.model_name == 'CNR':
        model = CNR_Anchor(args, data)
        i_model = CNR_Intervener(args, data).to(args.device)
    if args.model_name == 'NAR':
        args.out_loop = 1
        args.hidden_dim = 300
        model = NAR(args, data)

    model.args = args
    return evaluator, model, i_model


def init_logger(args):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")

    writer = None
    title = '{}_{}_{}_{}_O:{}_I:{}_drop_{}_emb_{}_hidden_{}'.format(
        args.dataset_str, args.model_name, args.lr, args.K, args.out_loop,
        args.epochs, args.dropout, args.emb_dim, args.hidden_dim)
    if 'CAR' in args.model_name:
        title += '_reg_adv_{}_adv_epoch_{}_eps_{}_reg_{}'.format(args.reg_adv, args.adv_epoch, args.eps, args.reg)

    if args.tb_log:
        writer = SummaryWriter(comment=title)
    if args.txt_log:
        writer = '{}/new_log/{}_{}.txt'.format(parent_path, title, dt_string)
    return writer


def prepare_training_data(args, data, batch_user, batch_item):
    batch_neg_item = torch.LongTensor(random.sample(range(data.item_num), batch_item.shape[0])).to(args.device)
    if 'CNR' in args.model_name and args.has_counter_result:
        gen_sample_bz = min(int(batch_user.shape[0] * data.gen_train_ratio), data.gen_train_users.shape[0])
        gen_users_num = data.gen_train_users.shape[0]
        gen_sample_idx = random.sample(range(gen_users_num), gen_sample_bz)
        return (torch.cat([batch_user,data.gen_train_users[gen_sample_idx]], dim=0),
                torch.cat([data.user_feature_all[batch_user], data.gen_train_users_feature[gen_sample_idx]], dim=0),
                torch.cat([batch_item, data.gen_train_pos_items[gen_sample_idx]], dim=0),
                torch.cat([data.item_feature_all[batch_item], data.item_feature_all[data.gen_train_pos_items[gen_sample_idx]]], dim=0),
                torch.cat([batch_neg_item, data.gen_train_neg_items[gen_sample_idx]], dim=0),
                torch.cat([data.item_feature_all[batch_neg_item], data.item_feature_all[data.gen_train_neg_items[gen_sample_idx]]], dim=0)
        )

    else:
        return (batch_user, data.user_feature_all[batch_user],
                batch_item, data.item_feature_all[batch_item],
                batch_neg_item, data.item_feature_all[batch_neg_item])

def save_model_func(args, model):
    save_folder = parent_path + '/saved_model'
    file_name = 'best_{}_{}.pkl'.format(args.model_name, args.dataset_str)
    save_path = '{}/{}'.format(save_folder,file_name)
    torch.save(model.state_dict(), save_path)


def vis_func(args, data, model, intervention):
    word_list = []
    for word, id in data.feature_id_dict.items():
        word_list.append(word)

    if intervention is not None:
        user_word_score_dict = intervention.vis(model)
    else:
        user_word_score_dict = model.vis(data)

    sorted_user_pertub_dict = defaultdict(list)
    word_sorted_dict = defaultdict(list)
    num_words = len(word_list)
    for user, score_v in user_word_score_dict.items():
        temp_dict = {}
        for word, score in zip(word_list, list(score_v)):
            temp_dict[word] = score

        word_sorted_dict[user] = sorted(temp_dict.items(), key=lambda x: x[0], reverse=False)

        if 'NAR' in args.model_name:
            temp_list = sorted(temp_dict.items(), key=lambda x: x[1], reverse=False)
            sorted_user_pertub_dict[user] = temp_list
        elif 'CNR' in args.model_name or 'CER' in  args.model_name:
            temp_list = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_user_pertub_dict[user] = temp_list
        elif 'CAR' in args.model_name:
            temp_list = sorted(temp_dict.items(), key=lambda x: abs(x[1]), reverse=False)
            sorted_user_pertub_dict[user] = temp_list
        elif 'counter' in args.model_name:
            temp_list = sorted(temp_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            sorted_user_pertub_dict[user] = temp_list

    vis_matrix = np.zeros((len(word_sorted_dict), num_words))
    for u_idx in range(len(word_sorted_dict)):
        vis_matrix[u_idx] = np.array([score for (w,score) in word_sorted_dict[u_idx]]) # word_sorted_dict[u_idx]

    with open(parent_path + '/saved_vis_matrix/{}_{}_vis_matrix.pkl'.format(args.dataset_str, args.model_name), 'wb') as file:
        pickle.dump(vis_matrix, file)

    user_feature_sentiment = data.user_feature_sentiment
    user_feature_sentiment_groundtruth = defaultdict()
    for uid, fea_sent in user_feature_sentiment.items():
        user_feature_sentiment_groundtruth[uid] = list(user_feature_sentiment[uid].keys())

    user_feature_perturb = defaultdict()
    for uid, fea_purb in sorted_user_pertub_dict.items():
        user_feature_perturb[uid] = [data.feature_id_dict[word] for word, _ in fea_purb]

    p, r, f1, ndcg = explanation_evaluate(user_feature_sentiment_groundtruth, user_feature_perturb, k=10)
    log_info = 'Evaluate Explanation (User sentiment oriented) --> Dataset: {} Model: {} precision: {:#.4g}, recall: ' \
          '{:#.4g}, f1: {:#.4g}, ndcg: {:#.4g}'.format(args.dataset_str, args.model_name, p, r, f1, ndcg)
    print(log_info)

    with open(parent_path+'/results/vis_result_product.csv', 'a+') as file:
        file.write('{},{},{},{}\n'.format(args.dataset_str, args.model_name, f1, ndcg))

    exit(1)


def explanation_evaluate(gt, pred, k=10):
    p_total, r_total, f1_total = list(), list(), list()
    ndcg_total = list()
    for uid, wids in gt.items():
        fit = pred[uid][:k]

        # precision, recall, f1
        cross = float(len([i for i in fit if i in wids]))
        p = cross / len(fit)
        r = cross / len(wids)
        if cross > 0:
            f1 = 2.0 * p * r / (p + r)
        else:
            f1 = 0.0

        p_total.append(p)
        r_total.append(r)
        f1_total.append(f1)

        # ndcg
        temp = 0
        Z_u = 0
        for j in range(min(len(fit), len(wids))):
            Z_u = Z_u + 1 / np.log2(j + 2)
        for j in range(len(fit)):
            if fit[j] in set(wids):
                temp = temp + 1 / np.log2(j + 2)

        if Z_u == 0:
            temp = 0
        else:
            temp = temp / Z_u
        ndcg_total.append(temp)

    return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f1_total).mean(), np.array(ndcg_total).mean()