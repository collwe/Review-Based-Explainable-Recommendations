import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP_predictor(num_output_layer, input_dim, hidden_dim, dropout=0.5):
    fc_pipe = None
    if num_output_layer == 1:
        fc_pipe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1)
        )
    elif num_output_layer == 2:
        fc_pipe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1)
        )
    elif num_output_layer == 3:
        fc_pipe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
    else:
        exit(1)
    return fc_pipe


class NCF(nn.Module):
    def __init__(self, args, data):
        super(NCF, self).__init__()
        self.args = args
        user_num = data.user_num
        item_num = data.item_num
        hidden_dim = args.hidden_dim
        self.embed_user = nn.Embedding(user_num, args.emb_dim)
        self.embed_item = nn.Embedding(item_num, args.emb_dim)
        input_dim = args.emb_dim*2
        self.fc_pipe = MLP_predictor(args.num_output_layer, input_dim, args.hidden_dim, args.dropout)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.embed_user.weight.requires_grad = True
        self.embed_item.weight.requires_grad = True
        self.drop = nn.Dropout(p=args.dropout)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.loss_fn = torch.nn.BCELoss()



    def forward(self, user, user_feat, item, item_feat):
        user_embedding = self.embed_user(user)
        item_embedding = self.embed_item(item)

        hidden = torch.cat((user_embedding, item_embedding), -1)
        if self.args.concat_dropout:
            hidden = self.drop(hidden)
        prediction = self.fc_pipe(hidden)
        return prediction.view(-1)


    def compute_loss(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch, epoch=None):
        # compute positive socre
        pos_score = F.sigmoid(self.forward(user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch))
        # compute negative socre
        neg_score = F.sigmoid(self.forward(user_batch, user_feature_batch, neg_item_batch, neg_item_feature_batch))
        # compute loss
        pred = torch.cat([pos_score, neg_score])
        label = torch.ones_like(pred)
        label[pos_score.shape[0]:] = 0
        loss = self.loss_fn(pred, label)
        return loss