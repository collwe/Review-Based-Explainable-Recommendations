import random
from collections import defaultdict
import torch.utils.data as tdata
import torch
import tqdm
import numpy as np
import torch.nn.functional as F

import os

from model_NCF import MLP_predictor

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


class CER(torch.nn.Module):
    def __init__(self, args, data):
        super(CER, self).__init__()
        self.args = args
        item_feat_dim = data.item_feat_dim
        input_dim = item_feat_dim * 2

        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)

        self.activation = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()


    def forward(self, user, user_feat, item, item_feat):
        if self.args.norm_feat:
            user_feat = F.normalize(user_feat, p=2, dim=1)
            item_feat = F.normalize(item_feat, p=2, dim=1)

        hidden = torch.cat([user_feat, item_feat], 1)
        if self.args.concat_dropout:
            hidden = self.drop(hidden)
        out = self.fc_pipe(hidden)
        out = self.activation(out)
        return out


    def vis(self, data): # generate pertubation dict
        # 1.load explanation generation args
        for param in self.parameters():
            param.requires_grad = False

        user_set = set(data.train_user_all.cpu().tolist())
        test_data = []
        for user in user_set:
            user_idx = torch.where(data.train_user_all == user)[0]
            pos_items = torch.tensor(random.choices(data.pos_item_all[user_idx], k=5)).to(data.train_user_all.device)
            neg_items = torch.tensor(random.choices(range(data.statistics['item_number']), k=10)).to(pos_items.device)
            items = torch.cat([pos_items,neg_items])
            test_data.append([user, items])
        data.test_data = test_data

        # Create optimization model
        opt_model = ExpOptimizationModel(base_model=self, rec_dataset=data, device=self.args.device, args=self.args)

        user_pertub_dict = opt_model.generate_explanation()
        return user_pertub_dict


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch, epoch=None):
        # compute positive socre
        pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
        # compute negative socre
        neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
        # compute loss
        pred = torch.cat([pos_score, neg_score])
        label = torch.ones_like(pred)
        label[pred.shape[0]//2:] = 0
        loss = self.loss_fn(pred, label)
        return loss



class ExpOptimizationModel(torch.nn.Module):
    def __init__(self, base_model, rec_dataset, device, args):
        super(ExpOptimizationModel, self).__init__()
        self.base_model = base_model
        self.rec_dataset = rec_dataset
        self.device = device
        self.args = args
        self.u_i_exp_dict = {}
        self.user_feature_matrix = self.rec_dataset.user_feature_all
        self.item_feature_matrix = self.rec_dataset.item_feature_all
        self.rec_dict, self.user_matrix, self.items_matrix = self.generate_rec_dict()


    def generate_test_data(self):
        for user, positive_items in self.train_user_positive_items_dict.items():
            assert len(positive_items) > 0
            for item in positive_items:
                pos_item_id = int(item)


    def generate_rec_dict(self):
        rec_dict = {}
        for row in self.rec_dataset.test_data:
            user = row[0]
            items = row[1]
            user_features = self.user_feature_matrix[user].repeat(len(items), 1)
            scores = self.base_model(None, user_features, None, self.item_feature_matrix[items]).squeeze()
            scores = np.array(scores.to('cpu'))
            sort_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            sorted_items = [items[i] for i in sort_index]
            rec_dict[user] = sorted_items

        user_list = []
        items_list = []
        for user, items in tqdm.tqdm(list(rec_dict.items())):
            user_list.append(user)
            items_list.append(items)
        user_matrix = torch.tensor(user_list)
        items_matrix = torch.tensor(items_list)
        return rec_dict, user_matrix, items_matrix


    def generate_explanation(self):
        num_user = len(self.user_matrix)
        b = 300
        idx_data = tdata.TensorDataset(torch.LongTensor(np.arange(num_user)).to(self.user_feature_matrix.device))
        dataloader = tdata.DataLoader(idx_data, batch_size=b, shuffle=False)
        delta_cache = torch.zeros_like(self.user_feature_matrix).to(self.user_feature_matrix.device)
        user_pertub_dict = dict()

        for batch in dataloader:
            batch_idx = batch[0]
            batch_users = self.user_matrix[batch_idx]
            batch_items = self.items_matrix[batch_idx]
            margin_batch_item = batch_items[:, self.args.rec_k]
            margin_score = self.base_model(None, self.user_feature_matrix[batch_users],
                                           None, self.item_feature_matrix[margin_batch_item]).squeeze()

            if self.args.user_mask:
                mask_vec = torch.where(self.user_feature_matrix[batch_users] > 0, 1., 0.)
            else:
                mask_vec = torch.ones((batch_idx.shape[0], self.rec_dataset.feature_num), device=self.device)


            for i in range(self.args.rec_k):
                items = batch_items[:, i]
                optimize_delta = self.explain(self.user_feature_matrix[batch_users],
                    self.item_feature_matrix[items], margin_score, mask_vec)
                delta_cache[batch_users] = delta_cache[batch_users]+optimize_delta
        delta_cache = -1 * delta_cache
        delta_cache = delta_cache/self.args.rec_k

        for i in  range(num_user):
            user_pertub_dict[i] = delta_cache[i].cpu().numpy()

        return user_pertub_dict


    def explain(self, user_feature, item_feature, margin_score, mask_vec):
        exp_generator = EXPGenerator(
            self.rec_dataset, self.base_model,
            user_feature, item_feature,
            margin_score, mask_vec,
            self.device, self.args).to(self.device)

        # optimization
        optimizer = torch.optim.SGD(exp_generator.parameters(), lr=self.args.vis_lr, weight_decay=0)
        exp_generator.train()
        score = exp_generator()
        bpr, l2, l1, loss = exp_generator.loss(score)
        lowest_loss = loss
        optimize_delta = exp_generator.delta.detach().to('cpu').numpy()
        for epoch in range(self.args.step):
            exp_generator.zero_grad()
            score = exp_generator()
            bpr, l2, l1, loss = exp_generator.loss(score)
            loss.backward()
            optimizer.step()
            if loss < lowest_loss:
                lowest_loss = loss
                optimize_delta = exp_generator.delta.detach()
        return optimize_delta



class EXPGenerator(torch.nn.Module):
    def __init__(self, rec_dataset, base_model, user_feature, item_feature, margin_score, mask_vec, device, args):
        super(EXPGenerator, self).__init__()
        self.rec_dataset = rec_dataset
        self.base_model = base_model
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.margin_score = margin_score
        self.mask_vec = mask_vec
        self.device = device
        self.args = args
        self.feature_range = [0, 5]  # hard coded, should be improved later
        self.delta_range = self.feature_range[1] - self.feature_range[0]  # the maximum feature value.
        self.delta = torch.nn.Parameter( torch.FloatTensor(self.user_feature.shape[0], self.user_feature.shape[1]).uniform_(-self.delta_range, 0))

    def get_masked_item_feature(self):
        item_feature_star = torch.clamp(
            (self.item_feature + torch.clamp((self.delta * self.mask_vec), -self.delta_range, 0)), self.feature_range[0], self.feature_range[1])
        return item_feature_star

    def forward(self):
        item_feature_star = self.get_masked_item_feature()
        score = self.base_model(None, self.user_feature, None, item_feature_star)
        return score

    def loss(self, score):
        bpr = torch.nn.functional.relu(self.args.alp + score.squeeze(-1) - self.margin_score) * self.args.lam
        l2 = torch.linalg.norm(self.delta, dim=-1)
        l1 = torch.linalg.norm(self.delta, ord=1, dim=-1) * self.args.gam
        loss = l2 + bpr + l1
        loss = loss.mean()
        return bpr, l2, l1, loss