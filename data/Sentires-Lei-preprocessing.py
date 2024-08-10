#!/usr/bin/env python
# coding: utf-8

import random
import pickle
import argparse
import numpy as np

from math import exp
from sklearn.decomposition import NMF
from collections import Counter, defaultdict

def parameter_parser():
    parser = argparse.ArgumentParser(description="Process the output from Sentire-Lei.")

    parser.add_argument("--path_input",                type=str,   default="../English-Jar/lei/input/",   help="input data path")
    parser.add_argument("--path_output",               type=str,   default="../English-Jar/lei/output/",  help="output data path")
    parser.add_argument("--file_input",                type=str,   default="product2json.pickle",  help="input file for sentire-lei")
    parser.add_argument("--file_output",               type=str,   default="reviews.pickle",       help="output file for sentire-lei")
    parser.add_argument("--rating_max",                type=int,   default=5,                      help="maximum rating in review")
    parser.add_argument("--n_sampled_evaluation",      type=int,   default=100,                    help="num of candidates items in the evaluation of test/val")
    parser.add_argument("--train_ratio",               type=float, default=0.8,                    help="train split ratio")
    parser.add_argument("--test_ratio",                type=float, default=0.1,                    help="test split ratio")
    parser.add_argument("--validation_ratio",          type=float, default=0.1,                    help="validation split ratio")
    parser.add_argument("--path_savefile",             type=str,   default="../pickle_output/",    help="file to save pickle output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parameter_parser()

    # Input #
    with open(args.path_input + args.file_input, 'rb') as f:
        input = pickle.load(f)
        
    # Output #
    with open(args.path_output + args.file_output, 'rb') as f:
        output = pickle.load(f)

    # Get all word features #
    fea_wordset = set()
    fea_wordlist = list()
    for res_tuple in output:
        if 'sentence' in res_tuple:
            user, item, rating = res_tuple['user'], res_tuple['item'], res_tuple['rating']
            sentence = res_tuple['sentence']
            for s in sentence:
                fea, value, location, sentiment = s[0], s[1], s[2], s[3]
                fea_wordset.add(fea)
                fea_wordlist.append(fea)
            
    counter = Counter(fea_wordlist)
    print('# of raw feature words is:%d', len(fea_wordset))
    fea_sorted = sorted(counter.items(), key=lambda item:item[1], reverse=True)
    print('The word features in sorted order are:', fea_sorted)

    #  ----- Get all user, items that satisfy: ----- #
    #       1. have word features 
    #       2. user-item has at least one positive interaction(ratings >= 4) assume ratings can be [1, 5]
    #  format:
    #       dictionary of dictionary
    #  example:
    #       users[user][feature]
    #       - user can be 'A16XRPF40679KG'
    #       - feature can be 'episode'

    users, items = defaultdict(defaultdict), defaultdict(defaultdict)
    user_pos_item_count = defaultdict(int)
    for res_tuple in output:
        if 'sentence' in res_tuple:
            user, item, rating = res_tuple['user'], res_tuple['item'], res_tuple['rating']
            
            if rating >= 4:
                users[user]
                items[item]
                sentence = res_tuple['sentence']
                for s in sentence:
                    fea, value, location, sentiment = s[0], s[1], s[2], s[3]

                    if fea not in users[user]:
                        users[user][fea] = 1
                    else:
                        users[user][fea] += 1

                    if fea not in items[item]:
                        items[item][fea] = (1, sentiment)
                    else:
                        prev_count, prev_sentiment = items[item][fea][0], items[item][fea][1]
                        items[item][fea] = (prev_count + 1, (prev_sentiment * prev_count + sentiment)/(prev_count + 1))


    # generate user_id_dict, item_id_dict, feature_id_dict
    user_ids, item_ids = list(users.keys()), list(items.keys())

    user_id_dict = {key:idx for idx, key in enumerate(user_ids)}
    item_id_dict = {key:idx for idx, key in enumerate(item_ids)}
    fea_sorted_word_only = [x[0] for x in fea_sorted]
    feature_id_dict = {key:idx for idx, key in enumerate(fea_sorted_word_only)}
    print('# user, # item, # fea are:', len(user_id_dict), len(item_id_dict), len(feature_id_dict))


    # ----- generate features: ID based dictionary ----- #
    # referece: Zhang, et al., Explicit Factor Models for Explainable Recommendation based on Phrase-level Sentiment Analysis, SIGIR 15

    u_fea_count = defaultdict()
    u_fea_smooth = defaultdict()
    u_fea_smooth_imputed = defaultdict()

    i_fea_count = defaultdict()
    i_fea_smooth = defaultdict()
    i_fea_smooth_imputed = defaultdict()

    for user in users:
        uid = user_id_dict[user]
        u_fea_count[uid] = [0] * len(feature_id_dict)
        u_fea_smooth[uid] = [0] * len(feature_id_dict)
        
        fea_hist = users[user]
        for fea in fea_hist:
            fidx = feature_id_dict[fea]
            fcount = fea_hist[fea]
            # raw count for 'u_fea_count'
            u_fea_count[uid][fidx] = fcount 
            
            # smooth score in range [1, 5] for 'u_fea_smooth'
            u_fea_smooth[uid][fidx] =  1 + (args.rating_max - 1) * (2/(1 + exp(-fcount)) - 1)

    for item in items:
        iid = item_id_dict[item]
        i_fea_count[iid] = [0] * len(feature_id_dict)
        i_fea_smooth[iid] = [0] * len(feature_id_dict)
        
        fea_sent_tuple = items[item]
        for fea in fea_sent_tuple:
            fidx = feature_id_dict[fea]
            fcount, avg_sentiment = fea_sent_tuple[fea]
            
            # raw count for 'i_fea_count'
            i_fea_count[iid][fidx] = fcount 
            
            # smooth score in range [1, 5] for 'i_fea_smooth'
            i_fea_smooth[iid][fidx] =  1 + (args.rating_max - 1)/(1 + exp(-fcount * avg_sentiment))

    # ----- impute values for u_fea_smooth and i_fea_smooth, ordere by user id dict ----- #
    u_fea_smooth_matrix = list()
    for idx in range(len(user_id_dict)):
        u_fea_smooth_matrix.append(u_fea_smooth[idx])
        
    i_fea_smooth_matrix = list()
    for idx in range(len(item_id_dict)):
        i_fea_smooth_matrix.append(i_fea_smooth[idx])

    model_user = NMF(n_components=16, init='random', random_state=0)
    W_user = model_user.fit_transform(np.array(u_fea_smooth_matrix))
    H_user = model_user.components_
    u_fea_smooth_matrix_imputed = np.matmul(W_user, H_user)

    model_item = NMF(n_components=16, init='random', random_state=0)
    W_item = model_item.fit_transform(np.array(i_fea_smooth_matrix))
    H_item = model_item.components_
    i_fea_smooth_matrix_imputed = np.matmul(W_item, H_item)

    for idx in range(len(u_fea_smooth_matrix_imputed)):
        u_fea_smooth_imputed[idx] = u_fea_smooth_matrix_imputed[idx]
        
    for idx in range(len(i_fea_smooth_matrix_imputed)):
        i_fea_smooth_imputed[idx] = i_fea_smooth_matrix_imputed[idx]


    # ----- get user's positive item list FOR both train and test ----- #
    num_items = len(item_id_dict)
    total_user_positive_items_dict = defaultdict(list)
    train_user_positive_items_dict = defaultdict(list) # train set: user's positive item list
    test_ground_truth_user_items_dict = defaultdict(list)   # test set:  user's positive item list
    test_user_items_dict = defaultdict(list) # test set:  user's sample evaluation item set
    validation_ground_truth_user_items_dict = defaultdict(list)   # validation set:  user's positive item list
    validation_user_items_dict = defaultdict(list) # validation set:  user's sample evaluation item set, 

    for res_tuple in output:
        if 'sentence' in res_tuple:
            user, item, rating = res_tuple['user'], res_tuple['item'], res_tuple['rating']
            if rating >= 4:
                uid = user_id_dict[user]
                iid = item_id_dict[item]
                
                total_user_positive_items_dict[uid].append(iid)

    for uid, pos_iid_list in total_user_positive_items_dict.items():
        num_pos = len(pos_iid_list)
        if num_pos <= 1:
            train_user_positive_items_dict[uid].extend(pos_iid_list)
        elif num_pos <= 5:
            # test ground truth dict (only one postive test item)
            rand_idx = random.randint(0, num_pos-1)
            test_iid = [pos_iid_list[rand_idx]] # one element list
            test_ground_truth_user_items_dict[uid].extend(test_iid)
            
            # test sampled evaluation dict (test GT iid + random sampled negative iid)
            candidate_iid = list( set(range(num_items)) - set(pos_iid_list) - set(test_iid) )
            test_user_items_dict[uid].extend(test_iid + random.sample(candidate_iid, args.n_sampled_evaluation - 1))
            
            # train ground truth dict
            train_iid = pos_iid_list[:rand_idx] + pos_iid_list[rand_idx+1:] 
            train_user_positive_items_dict[uid].extend(train_iid)
        else:
            n_train = round(num_pos * args.train_ratio)
            n_validation = round(num_pos * args.validation_ratio)
            n_test = num_pos - n_train - n_validation
            
            # test ground truth dict (10% test, 10% validation, 80% train)
            rand_idx = random.sample(range(num_pos))
            train_idx = rand_idx[:n_train]
            validation_idx = rand_idx[n_train:n_train + n_validation]
            test_idx = rand_idx[n_train + n_validation:]
            
            train_iid = [pos_iid_list[idx] for idx in train_idx]
            validation_iid = [pos_iid_list[idx] for idx in validation_idx]
            test_iid = [pos_iid_list[idx] for idx in test_idx]

            test_ground_truth_user_items_dict[uid].extend(test_iid)
            validation_ground_truth_user_items_dict[uid].extend(validation_iid)
            
            # test sampled evaluation dict (test GT iid + random sampled negative iid)
            candidate_iid = list( set(range(num_items)) - set(pos_iid_list) - set(test_iid) - set(validation_iid) )
            test_user_items_dict[uid].extend(test_iid + random.sample(candidate_iid, args.n_sampled_evaluation - n_test))
            validation_user_items_dict[uid].extend(test_iid + random.sample(candidate_iid, args.n_sampled_evaluation - n_validation))

            # train ground truth dict
            train_iid = [pos_iid_list[x] for x in range(num_pos) if x in train_idx]
            train_user_positive_items_dict[uid].extend(train_iid)

    # ----- generate the required input file -----
    pickle.dump(user_id_dict, open(args.path_savefile + 'user_id_dict', 'wb'))
    pickle.dump(item_id_dict, open(args.path_savefile + 'item_id_dict', 'wb'))
    pickle.dump(feature_id_dict, open(args.path_savefile + 'feature_id_dict', 'wb'))

    pickle.dump(u_fea_count, open(args.path_savefile + 'u_fea_count', 'wb'))
    pickle.dump(u_fea_smooth, open(args.path_savefile + 'u_fea_smooth', 'wb'))
    pickle.dump(u_fea_smooth_imputed, open(args.path_savefile + 'u_fea_smooth_imputed', 'wb'))
    pickle.dump(i_fea_count, open(args.path_savefile + 'i_fea_count', 'wb'))
    pickle.dump(i_fea_smooth, open(args.path_savefile + 'i_fea_smooth', 'wb'))
    pickle.dump(i_fea_smooth_imputed, open(args.path_savefile + 'i_fea_smooth_imputed', 'wb'))

    pickle.dump(train_user_positive_items_dict, open(args.path_savefile + 'train_user_positive_items_dict', 'wb'))
    pickle.dump(test_ground_truth_user_items_dict, open(args.path_savefile + 'test_ground_truth_user_items_dict', 'wb'))
    pickle.dump(test_user_items_dict, open(args.path_savefile + 'test_compute_user_items_dict', 'wb'))
    pickle.dump(test_ground_truth_user_items_dict, open(args.path_savefile + 'test_ground_truth_user_items_dict', 'wb'))

    pickle.dump(validation_user_items_dict, open(args.path_savefile + 'validation_compute_user_items_dict', 'wb'))
    pickle.dump(validation_ground_truth_user_items_dict, open(args.path_savefile + 'validation_ground_truth_user_items_dict', 'wb'))

