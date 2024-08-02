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


class CAR(nn.Module):
    def __init__(self, args, data):
        super(CAR, self).__init__()
        self.args = args
        user_num = data.user_num
        item_num = data.item_num
        self.user_size = data.user_feat_dim
        self.item_size = data.item_feat_dim
        self.reg = args.reg
        self.reg_adv = args.reg_adv
        self.eps = args.eps

        if args.norm_feat:
            user_feature_all = F.normalize(data.user_feature_all, p=2, dim=1)
            item_feature_all = F.normalize(data.item_feature_all, p=2, dim=1)
        else:
            user_feature_all = data.user_feature_all
            item_feature_all = data.item_feature_all

        self.feat_embed_user = nn.Embedding(user_num, data.item_feat_dim, _weight=user_feature_all)
        self.feat_embed_item = nn.Embedding(item_num, data.item_feat_dim, _weight=item_feature_all)
        self.feat_embed_user.weight.requires_grad = True
        self.feat_embed_item.weight.requires_grad = True

        self.user_feat_linear = nn.Linear(data.user_feat_dim, args.hidden_dim, bias=False)
        self.item_feat_linear = nn.Linear(data.item_feat_dim, args.hidden_dim, bias=False)

        input_dim = args.hidden_dim

        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, user, user_feat, item, item_feat, return_all=False):
        u_feat = self.feat_embed_user(user)
        i_feat = self.feat_embed_item(item)

        u_feat = F.dropout(F.relu(self.user_feat_linear(u_feat)), p=self.args.dropout)
        i_feat = F.dropout(F.relu(self.item_feat_linear(i_feat)), p=self.args.dropout)

        if self.args.num_output_layer:
            pos_embedding = u_feat * i_feat
            if self.args.concat_dropout:
                pos_embedding = self.drop(pos_embedding)
            x_ui = self.fc_pipe(pos_embedding)
        else:
            x_ui = (u_feat * i_feat).sum(dim=1)
        if return_all:
            return x_ui, u_feat, i_feat
        else:
            return x_ui


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch,
                                                            neg_item_batch, neg_item_feature_batch, epoch=None):
        x_ui, u_feat, i_feat = self.forward(user_batch, user_feature_batch, pos_item_batch,
                                                                pos_item_feature_batch, return_all=True)
        x_uj, u_feat, j_feat = self.forward(user_batch, user_feature_batch, neg_item_batch,
                                                                neg_item_feature_batch, return_all=True)

        x_uij = torch.clamp(x_ui - x_uj, min=-80.0, max=1e8)
        log_prob = F.logsigmoid(x_uij).sum()
        loss = -log_prob

        if epoch in range(0, self.args.epochs - self.args.adv_epoch):
            return loss
        else:
            u_feat.retain_grad()
            i_feat.retain_grad()
            j_feat.retain_grad()
            # Backward to get grads
            loss.backward(retain_graph=True)
            grad_u = u_feat.grad.detach().clone()
            grad_i = i_feat.grad.detach().clone()
            grad_j = j_feat.grad.detach().clone()
            # Construct adversarial perturbation
            delta_u = nn.functional.normalize(grad_u, p=2, dim=1, eps=self.eps)
            delta_i = nn.functional.normalize(grad_i, p=2, dim=1, eps=self.eps)
            delta_j = nn.functional.normalize(grad_j, p=2, dim=1, eps=self.eps)
            # Add adversarial perturbation to embeddings
            u_feat = u_feat + delta_u
            i_feat = i_feat + delta_i
            j_feat = j_feat + delta_j

            if self.args.num_output_layer:
                pos_embedding_ = torch.cat([u_feat, i_feat], dim=-1)
                neg_embedding_ = torch.cat([u_feat, j_feat], dim=-1)
                if self.args.concat_dropout:
                    pos_embedding_ = self.drop(pos_embedding_)
                    neg_embedding_ = self.drop(neg_embedding_)
                x_ui_adv = self.fc_pipe(pos_embedding_)
                x_uj_adv = self.fc_pipe(neg_embedding_)
            else:
                x_ui_adv = (u_feat * i_feat).sum(dim=1)
                x_uj_adv = (u_feat * j_feat).sum(dim=1)
            # Calculate APR loss
            x_uij_adv = torch.clamp(x_ui_adv - x_uj_adv, min=-80.0, max=1e8)
            log_prob = F.logsigmoid(x_uij_adv).sum()
            adv_loss = self.reg_adv * (-log_prob) + loss

            return adv_loss


    def vis(self, data):
        train_users = data.train_user_all
        train_pos_items = data.pos_item_all
        train_neg_items = torch.tensor(random.choices(range(data.statistics['item_number']),
                                                k=train_pos_items.shape[0])).to(train_pos_items.device)
        user_pertub_dict = defaultdict(list)

        for _ in range(50):
            for i in range(len(train_users)):
                self.zero_grad()
                user_batch = train_users[i]
                pos_item_batch = train_pos_items[i]
                neg_item_batch = train_neg_items[i]

                user_id_t = torch.LongTensor([user_batch]).to(self.args.device)
                pos_item_id_t = torch.LongTensor([pos_item_batch]).to(self.args.device)
                neg_item_id_t = torch.LongTensor([neg_item_batch]).to(self.args.device)


                u_feat = self.feat_embed_user(user_id_t)
                i_feat = self.feat_embed_item(pos_item_id_t)
                j_feat = self.feat_embed_item(neg_item_id_t)
                if self.args.norm_feat:
                    u_feat = F.normalize(u_feat, p=2, dim=1)
                    i_feat = F.normalize(i_feat, p=2, dim=1)
                    j_feat = F.normalize(j_feat, p=2, dim=1)

                u_feat = F.relu(self.user_feat_linear(u_feat))
                i_feat = F.relu(self.item_feat_linear(i_feat))
                j_feat = F.relu(self.item_feat_linear(j_feat))

                if self.args.num_output_layer:
                    pos_embedding_ = u_feat * i_feat
                    neg_embedding_ = u_feat * j_feat
                    if self.args.concat_dropout:
                        pos_embedding_ = self.drop(pos_embedding_)
                        neg_embedding_ = self.drop(neg_embedding_)
                    x_ui = self.fc_pipe(pos_embedding_)
                    x_uj = self.fc_pipe(neg_embedding_)
                else:
                    x_ui = (u_feat * i_feat).sum(dim=1)
                    x_uj = (u_feat * j_feat).sum(dim=1)

                x_uij = torch.clamp(x_ui - x_uj, min=-80.0, max=1e8)
                log_prob = F.logsigmoid(x_uij).sum()
                loss = -log_prob

                u_feat.retain_grad()
                i_feat.retain_grad()
                j_feat.retain_grad()

                loss.backward(retain_graph=True)
                grad_u = u_feat.grad.detach().clone()

                pertub_on_u = nn.functional.normalize(grad_u, p=2, dim=1, eps=self.eps)
                user_pertub_dict[user_batch.item()].append( pertub_on_u.reshape(1,-1).detach().cpu().numpy() )


        for k,v in user_pertub_dict.items():
            if len(v) > 1:
                v = np.concatenate(v,axis=0).mean(axis=0)
                user_pertub_dict[k] = v
            else:
                user_pertub_dict[k] = v[0].reshape(-1,)

        return user_pertub_dict