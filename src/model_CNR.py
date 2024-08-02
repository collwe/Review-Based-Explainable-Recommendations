import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_NCF import MLP_predictor


class CNR_Anchor(nn.Module):
# data.user_feat_dim, data.item_feat_dim
    def __init__(self, args, data):
        super(CNR_Anchor, self).__init__()
        print('args: ', args)
        self._temp = 0.1
        self._cnt = 0
        self.args = args
        self.data = data
        hidden_dim = args.hidden_dim
        self.user_number = data.statistics['user_number']
        self.item_number = data.statistics['item_number']
        self.feature_number = data.item_feat_dim

        self.user_map = nn.Linear(self.data.user_feat_dim, args.hidden_dim, bias=False)
        self.item_map = nn.Linear(self.data.item_feat_dim, args.hidden_dim, bias=False)

        self.bi_cross_entropy = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        input_dim = self.args.hidden_dim
        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)

        self.drop = nn.Dropout(p=args.dropout)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch,
                     neg_item_batch, neg_item_feature_batch, epoch):
        # compute positive socre
        pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
        # compute negative socre
        neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
        loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        return loss


    def forward(self, users, user_features, items, item_features):
        user_mapped = F.dropout(F.relu(self.user_map(user_features)), p=self.args.dropout)
        item_mapped = F.dropout(F.relu(self.item_map(item_features)), p=self.args.dropout)
        ui_feature = user_mapped * item_mapped
        if self.args.concat_dropout:
            ui_feature = self.drop(ui_feature)
        result = self.fc_pipe(ui_feature)
        return result




class CNR_Intervener(nn.Module):
    def __init__(self, args, data):
        super(CNR_Intervener, self).__init__()
        print('args: ', args)
        self.args = args
        self.device = args.device
        self.data = data
        self.user_number = data.statistics['user_number']
        self.item_number = data.statistics['item_number']
        self.feature_number = data.item_feat_dim # data.statistics['feature_number']
        self.bi_cross_entropy = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def reset(self, bz):
        self.tau = nn.Parameter(torch.empty(bz, self.feature_number, device=self.device),  requires_grad=True)
        torch.nn.init.uniform_(self.tau, a=-1, b=1)
        self.optimizer = torch.optim.Adam([self.tau], lr=self.args.intv_lr, weight_decay=0.0)


    def set_anchor(self, anchor, grad):
        self.anchor = anchor
        self.anchor.eval()
        # requires_grad set to False
        for name, param in self.anchor.named_parameters():
            param.requires_grad = grad


    def forward(self, anchor, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch,
                neg_item_feature_batch, fid=-1):

        if not self.args.intervener_soft:
            self.mask = torch.zeros(self.tau.shape[0], self.feature_number).to(self.device)
            if (fid == -1):
                index = torch.topk(user_feature_batch, self.args.intervener_feature_number, dim=1)[1]
            else:
                index = torch.ones(self.args.intervener_batch_size, 1).long().cuda() * fid
            self.mask.scatter_(1, index, 1.)
            self.masked_tau = torch.mul(self.tau, self.mask)
        else:
            self.masked_tau = self.tau

        # generate new feature
        user_feature_t = torch.add(user_feature_batch, self.masked_tau)
        pos_score = anchor.forward(user_batch, user_feature_t, pos_item_batch, pos_item_feature_batch)
        neg_score = anchor.forward(user_batch, user_feature_t, neg_item_batch, neg_item_feature_batch)
        conf = - torch.nn.functional.logsigmoid(neg_score - pos_score)
        loss_sum = conf.sum()

        loss_sum += self.args.intervener_reg * torch.norm(self.masked_tau, 2)
        if self.args.intervener_soft:
            loss_sum += self.args.intervener_l1_reg * torch.norm(self.masked_tau, 1)

        return loss_sum, conf # conf > 0 and conf < 0.693 is nice


    def counter_data_generation_batch(self, model, user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature):
        for i in range(self.args.intervener_iteration):
            self.optimizer.zero_grad()
            loss, conf = self.forward(model, user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature,
                                 self.args.case_model)
            loss.backward()
            model.zero_grad()
            self.optimizer.step()
        self.optimizer.zero_grad()

        loss, conf = self.forward(model, user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature,
                                self.args.case_model)
        final_tau = self.masked_tau
        final_conf = conf

        return final_tau, final_conf


    def counter_data_generation(self, model):
        gen_train_users = []
        gen_train_users_feature = []
        gen_train_pos_items = []
        gen_train_neg_items = []

        for batch_user, batch_item in self.data.train_dataloader:
            self.reset(batch_item.shape[0])
            batch_neg_item = torch.LongTensor(random.sample(range(self.data.item_num), batch_item.shape[0])).to(self.args.device)
            tau, conf = self.counter_data_generation_batch(model, batch_user, self.data.user_feature_all[batch_user],
                                               batch_item, self.data.item_feature_all[batch_item],
                                               batch_neg_item, self.data.item_feature_all[batch_neg_item])
            tau = tau.detach()
            conf = conf.detach()
            filter = conf<self.args.confidence
            filter = filter.reshape(-1)
            gen_train_users.append(batch_user[filter])
            gen_train_users_feature.append(self.data.user_feature_all[batch_user][filter] + tau[filter])
            gen_train_pos_items.append(batch_neg_item[filter])
            gen_train_neg_items.append(batch_item[filter])

        return torch.cat(gen_train_users, dim=0), torch.cat(gen_train_users_feature, dim=0), \
               torch.cat(gen_train_pos_items, dim=0), torch.cat(gen_train_neg_items, dim=0)


    def vis(self, model):
        tau_matrix = torch.zeros(self.data.train_user_all.shape[0], self.data.user_feat_dim)
        counter_vec = torch.zeros(self.data.train_user_all.shape[0], 1)
        for batch_user, batch_item in self.data.train_dataloader:
            self.reset(batch_item.shape[0])
            batch_neg_item = torch.LongTensor(random.sample(range(self.data.item_num), batch_item.shape[0])).to(self.args.device)
            tau, conf = self.counter_data_generation_batch(model,
                                               batch_user, self.data.user_feature_all[batch_user],
                                               batch_item, self.data.item_feature_all[batch_item],
                                               batch_neg_item, self.data.item_feature_all[batch_neg_item])
            tau = tau.detach().cpu()
            tau_matrix[batch_user] += tau
            counter_vec[batch_user] += 1
            conf = conf.detach()
        counter_vec += 1e-9
        tau_matrix = tau_matrix / counter_vec
        user_att_dict = {}
        train_users = list(self.data.train_user_all.cpu().numpy())
        for idx, user in enumerate(train_users):
            user_att_dict[user] = tau_matrix[idx].numpy()

        return user_att_dict