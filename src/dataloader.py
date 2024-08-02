import os
import time

import numpy as np
import pickle

import torch
import torch.utils.data as tdata
from tqdm import tqdm

from src.utils import load_word_embedding

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


folder_dict = {}
folder_dict['video'] = 'Amazon_Instant_Video/'
folder_dict['Beauty'] = 'Beauty/'
folder_dict['Clothing'] = 'Clothing_Shoes_and_Jewelry/'
folder_dict['Music'] = 'Digital_Music/'
folder_dict['Musical'] = 'Musical_Instruments/'


class DataLoader_New():
    def __init__(self, args):
        self.args = args
        self.path = args.data_path + folder_dict[args.dataset_str]
        self.statistics = dict()
        user_id_dict = pickle.load(open(self.path + "user_id_dict", "rb"))
        item_id_dict = pickle.load(open(self.path + "item_id_dict", "rb"))
        self.feature_id_dict = pickle.load(open(self.path + "feature_id_dict", "rb"))

        self.statistics['user_number'] = len(user_id_dict.keys())
        self.statistics['item_number'] = len(item_id_dict.keys())
        self.statistics['feature_number'] = len(self.feature_id_dict.keys())
        self.user_num = self.statistics['user_number']
        self.item_num = self.statistics['item_number']

        self.user_feature_attention = pickle.load(open(self.path + "u_fea_{}".format(args.user_feat_type), "rb"))
        self.item_feature_quality = pickle.load(open(self.path + "i_fea_{}".format(args.user_feat_type), "rb"))
        self.train_user_positive_items_dict = pickle.load(open(self.path + "train_user_positive_items_dict", "rb"))
        self.ground_truth_user_items_dict = pickle.load(open(self.path + "test_ground_truth_user_items_dict", "rb"))
        self.compute_user_items_dict = pickle.load(open(self.path + "test_compute_user_items_dict", "rb"))
        self.user_feature_sentiment = pickle.load(open(self.path + "user_feature_sentiment", "rb"))

        self.w2v_feat_np = None
        self.init_data()
        self.generate_validation_corpus()
        args.intervener_feature_number = int(self.item_feat_dim * 0.6)
        print("Feature Dim:{}".format(self.item_feat_dim))


    def init_data(self):
        self.user_all = []
        self.item_all = []
        self.user_feature_all = []
        self.item_feature_all = []
        self.pos_item_all = []
        self.train_user_all = []

        # read data to tensor
        for i in range(len(self.user_feature_attention)):
            self.user_all.append(i)
            assert i in self.user_feature_attention
            self.user_feature_all.append(self.user_feature_attention[i])

        for i in range(len(self.item_feature_quality)):
            assert i in self.item_feature_quality
            self.item_feature_all.append(self.item_feature_quality[i])

        self.user_feature_all = torch.FloatTensor(self.user_feature_all).to(self.args.device)
        self.item_feature_all = torch.FloatTensor(self.item_feature_all).to(self.args.device)

        # pruning
        assert self.user_feature_all.shape[1] == self.item_feature_all.shape[1]
        DIM=self.args.feat_selection_num
        if self.user_feature_all.shape[1] > DIM:
            user_var_dims = torch.var(self.user_feature_all, dim=0)
            _, user_feat_filter_idx = torch.topk(user_var_dims, DIM)
            self.user_feature_all = self.user_feature_all[:, user_feat_filter_idx]
            self.user_feat_filter_idx = user_feat_filter_idx
            item_var_dims = torch.var(self.item_feature_all, dim=0)
            _, item_feat_filter_idx = torch.topk(item_var_dims, DIM)
            self.item_feature_all = self.item_feature_all[:, item_feat_filter_idx]
            self.item_feat_filter_idx = item_feat_filter_idx
            item_var_dims_set = set(list(item_feat_filter_idx.cpu().numpy()))

            # create filtered word dict
            filtered_feature_id_dict = {}
            for idx, (k,v) in enumerate(self.feature_id_dict.items()):
                if idx in item_var_dims_set:
                    filtered_feature_id_dict[k] = v
            self.feature_id_dict = filtered_feature_id_dict
        else:
            item_var_dims_set = set(range(self.user_feature_all.shape[1]))
            self.user_feat_filter_idx = torch.tensor(range(self.user_feature_all.shape[1])).to(self.args.device)
            self.item_feat_filter_idx = torch.tensor(range(self.item_feature_all.shape[1])).to(self.args.device)

        self.item_feat_dim = self.item_feature_all.shape[1]
        self.user_feat_dim = self.user_feature_all.shape[1]

        if 'NAR' in self.args.model_name:
            print("load word to vec data.")
            w2v_feat = []
            t = time.time()
            w2v_dict, w2v_matrix = load_word_embedding(debug=False)
            for word, id in tqdm(self.feature_id_dict.items()):
                if id not in item_var_dims_set:
                    print("id:{} not in dict".format(id))
                    exit(1)
                    continue
                if word in w2v_dict:
                    idx = w2v_dict[word]
                    w2v_feat.append(w2v_matrix[idx])
                elif ' ' in word:
                    tokens = word.split(' ')
                    temp = np.zeros(300, )
                    for token in w2v_dict:
                        if tokens in w2v_matrix:
                            temp += w2v_matrix[w2v_dict[token]]
                    w2v_feat.append(temp)
                else:
                    print('{} not in dict!'.format(word))
                    w2v_feat.append(np.random.random(300, ))
            self.w2v_feat_np = np.stack(w2v_feat, axis=0)
            print("Process w2v time:{}".format(time.time() - t))

        # create loader
        print("create train loader.")
        for user, positive_items in self.train_user_positive_items_dict.items():
            assert len(positive_items) > 0
            for item in positive_items:
                pos_item_id = int(item)
                self.pos_item_all.append(pos_item_id)
                self.train_user_all.append(user)
        self.pos_item_all = torch.LongTensor(self.pos_item_all).to(self.args.device)
        self.train_user_all = torch.LongTensor(self.train_user_all).to(self.args.device)

        self.train_dataloader = tdata.DataLoader(tdata.TensorDataset(self.train_user_all, self.pos_item_all),
                                                batch_size=self.args.batch_size, shuffle=True)


    def generate_validation_corpus(self):
        self.compute_user_items_feature_dict = dict()
        for user, item_list in self.compute_user_items_dict.items():
            tmp = []
            for item in item_list:
                tmp.append(self.item_feature_all[item].reshape(1,-1))
            self.compute_user_items_feature_dict[user] = tmp