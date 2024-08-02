import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_NCF import MLP_predictor


class NAR(nn.Module):
    def __init__(self, args, data):
        super(NAR, self).__init__()
        self.args = args
        user_num = data.user_num
        item_num = data.item_num
        w2v_feat = data.w2v_feat_np
        item_feat_dim = data.user_feat_dim
        emb_dim = args.emb_dim
        hidden_dim = args.hidden_dim
        self.embed_user = nn.Embedding(user_num, emb_dim)
        self.embed_item = nn.Embedding(item_num, emb_dim)
        self.q_user_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.q_item_proj = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.word_feat = torch.FloatTensor(w2v_feat)
        self.word_feat = self.word_feat.to(args.device)
        input_dim = item_feat_dim

        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)
        self.drop = nn.Dropout(p=args.dropout)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.embed_user.weight.requires_grad = True
        self.embed_item.weight.requires_grad = True
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)



    def scaled_dot_product(self, input_embed, mask=None):
        q, k, v = self.q_proj(input_embed), self.k_proj(input_embed), self.v_proj(input_embed)
        d_k = q.size()[0]
        attn_logits = torch.matmul(q.transpose(-2, -1), k)
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(v, attention)

        return values, attention


    def forward(self, user, user_feat, item, item_feat):
        if self.args.norm_feat:
            user_feat = F.normalize(user_feat, p=2, dim=1)
            item_feat = F.normalize(item_feat, p=2, dim=1)

        user_embedding = self.embed_user(user)
        item_embedding = self.embed_item(item)

        user_embedding = self.q_user_proj(user_embedding) # 32, 300
        item_embedding = self.q_item_proj(item_embedding)
        user_word_feat_att = torch.mm(user_embedding, self.word_feat.t())
        item_word_feat_att = torch.mm(item_embedding, self.word_feat.t())

        output_hidden = torch.cat([user_word_feat_att*user_feat, item_word_feat_att*item_feat], dim=-1)
        if self.args.concat_dropout:
            output_hidden = self.drop(output_hidden)
        output = self.fc_pipe(output_hidden)
        return output


    def vis(self, data):
        train_users = data.train_user_all
        train_users = torch.tensor(train_users).to(self.args.device)
        user_embedding = self.embed_user(train_users)
        user_embedding = self.q_user_proj(user_embedding)
        user_word_feat_att = torch.mm(user_embedding, self.word_feat.t())
        user_fused_att = user_word_feat_att.detach().cpu().numpy()

        user_att_dict = defaultdict(list)
        for idx, user in enumerate(train_users):
            user_att_dict[user.item()] = user_fused_att[idx]

        return user_att_dict


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch,
                                                                neg_item_batch, neg_item_feature_batch, epoch=None):
        # compute positive socre
        pos_score = self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch)
        # compute negative socre
        neg_score = self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch)
        # compute loss
        loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        return loss