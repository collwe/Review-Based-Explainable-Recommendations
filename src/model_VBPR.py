import math
import pickle
import random
import time
from collections import defaultdict

import torch
import numpy as np
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model_NCF import MLP_predictor


class VBPR(nn.Module):
    def __init__(self, args, data):
        super(VBPR, self).__init__()
        self.args = args
        user_num = data.user_num
        item_num = data.item_num
        emb_dim = args.emb_dim
        hidden_dim = args.hidden_dim
        user_feat_dim = data.user_feat_dim
        item_feat_dim = data.item_feat_dim

        self.embed_user = nn.Embedding(user_num, emb_dim)
        self.embed_item = nn.Embedding(item_num, emb_dim)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.embed_user.weight.requires_grad = True
        self.embed_item.weight.requires_grad = True

        self.user_linear = nn.Linear(user_feat_dim, hidden_dim)
        self.item_linear = nn.Linear(item_feat_dim, hidden_dim)

        input_dim = emb_dim + hidden_dim
        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)
        self.drop = nn.Dropout(p=args.dropout)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, user, user_feat, item, item_feat):
        if self.args.norm_feat:
            user_feat = F.normalize(user_feat, p=2, dim=1)
            item_feat = F.normalize(item_feat, p=2, dim=1)

        user_feat = F.relu(self.user_linear(user_feat))
        item_feat = F.relu(self.item_linear(item_feat))
        user_embedding = self.embed_user(user)
        item_embedding = self.embed_item(item)

        if self.args.num_output_layer:
            hidden = torch.cat([user_feat * item_feat, user_embedding * item_embedding], dim=-1)
            if self.args.concat_dropout:
                hidden = self.drop(hidden)
            output = self.fc_pipe(hidden)
        else:
            user_factor = torch.cat([user_feat, user_embedding], dim=1)
            item_factor = torch.cat([item_feat, item_embedding], dim=1)
            user_factor = self.drop(user_factor)
            item_factor = self.drop(item_factor)
            output = (user_factor * item_factor).sum(dim=1)

        return output


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch, epoch):
        # compute positive socre
        pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
        # compute negative socre
        neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
        # compute loss
        loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        return loss