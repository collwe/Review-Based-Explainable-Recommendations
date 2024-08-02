import argparse
import os
import random
import time

import torch
from tqdm import tqdm

from src.dataloader import DataLoader_New
from src.utils import set_seed, Evaluate, save_model_func, vis_func, init_model, prepare_training_data, init_logger

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))




def train(args, data, model, optimizer, epoch):
    loss_total = 0
    model.train()
    optimizer.zero_grad()

    for batch_user, batch_item in data.train_dataloader:
        data_cache = prepare_training_data(args, data, batch_user, batch_item)
        loss = model.compute_loss(*data_cache, epoch)
        optimizer.zero_grad()
        loss.backward()
        if 'CAR' in args.model_name:
            model.feat_embed_user.weight.grad = torch.zeros_like(model.feat_embed_user.weight.grad)
            model.feat_embed_item.weight.grad = torch.zeros_like(model.feat_embed_item.weight.grad)
        loss_total += loss.item()
        optimizer.step()

    return loss_total


def test(data, model, evaluator):
    pred = dict()
    ground_truth = dict()
    model.eval()
    test_user_list = list(data.compute_user_items_dict.keys())
    sampled_test_user_list = random.sample(test_user_list, min(5000, len(test_user_list)))
    print("testing...")
    test_t = time.time()
    with torch.no_grad():
        for u in tqdm(sampled_test_user_list):
            items = data.compute_user_items_dict[u]
            features = data.compute_user_items_feature_dict[u]
            u_extend = [u] * len(items)
            u_feature_extend = data.user_feature_all[u].repeat(len(items),1) # [data.user_feature_all[u].cpu().numpy()] * len(items)

            user_id_t = torch.LongTensor(u_extend).to(model.args.device)
            user_feature_t = u_feature_extend #torch.FloatTensor(u_feature_extend).to(model.args.device)
            item_id_t = torch.LongTensor(items).to(model.args.device)
            item_feature_t = torch.cat(features, dim=0) # torch.FloatTensor(features).to(model.args.device)

            scores = model.forward(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
            pred[u] = dict(zip(items, scores))
            ground_truth[u] = data.ground_truth_user_items_dict[u]

    map, mrr, p, r, f1, hit, ndcg = evaluator.evaluate(ground_truth, pred)
    print('Test used time:{}'.format(time.time() - test_t))
    return map, mrr, p, r, f1, hit, ndcg


def warmup(args, data, model, optimizer, evaluator):
    best_ndcg, best_test, best_epoch = 0, '', 0
    best_map = 0
    for epoch in tqdm(range(args.warmup_epochs)):
        t = time.time()
        print("data time:{}".format(time.time() - t))
        if epoch % args.test_interval == 0:
            map, mrr, p, r, f1, hit, ndcg = test(data, model, evaluator)

            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_test = 'P:{:#.4g} & R:{:#.4g} & f1:{:#.4g} & hit:{:#.4g} & ndcg:{:#.4g} & mrr:{:#.4g}'.format(p, r, f1, hit, ndcg, mrr)
                print("[Warm Up \t{}]".format(best_test))

        t = time.time()
        loss_total = train(args, data, model, optimizer, epoch)
        print("train time:{}".format(time.time() - t))
        print("Train:{}, loss:{}".format(epoch, loss_total))

    return best_epoch, best_test, best_ndcg


def run(args, data, model, optimizer, evaluator, writer=None, is_save_model=False, intervention=None):
    best_ndcg, best_test, best_epoch  = 0, '', 0
    best_map = 0
    if intervention is not None:
        model.eval()
        intervention.set_anchor(model, False)
        data.gen_train_users, data.gen_train_users_feature, data.gen_train_pos_items, data.gen_train_neg_items \
            = intervention.counter_data_generation(model)
        data.gen_train_ratio = data.gen_train_users.shape[0]/data.train_user_all.shape[0]
        args.has_counter_result = 1
        model.train()
        intervention.set_anchor(model, True)

    for epoch in tqdm(range(args.epochs)):
        # [[ Test Stage ]]
        t = time.time()
        print("data time:{}".format(time.time() - t))
        if epoch % args.test_interval == 0:
            map, mrr, p, r, f1, hit, ndcg = test(data, model, evaluator)
            print('[Test] Epoch:{} & p:{:#.4g} & r:{:#.4g} & f1:{:#.4g} & hit:{:#.4g}  & ndcg:{:#.4g}  & mrr:{:#.4g}.'.format(epoch, p, r, f1, hit, ndcg, mrr))
            if args.txt_log and writer is not None:
                with open(writer, 'a+') as file:
                    file.write('[Test] Epoch:{}  & p:{:#.4g}  & r:{:#.4g}  & f1:{:#.4g}  & hit:{:#.4g}  & ndcg:{:#.4g}  & mrr:{:#.4g}.\n'.format(epoch, p, r, f1, hit, ndcg, mrr))
            if args.tb_log and writer is not None:
                writer.add_scalar('Test/P', p, epoch)
                writer.add_scalar('Test/R', r, epoch)
                writer.add_scalar('Test/F1', f1, epoch)
                writer.add_scalar('Test/Hit', hit, epoch)
                writer.add_scalar('Test/NDCG', ndcg, epoch)

            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_test = '{:#.4g} & {:#.4g} & {:#.4g} & {:#.4g} & {:#.4g} & {:#.4g}'.format(p, r, f1, hit, ndcg, mrr)
                if is_save_model:
                    save_model_func(args, model)

        # [[ Train Stage ]]
        t = time.time()
        loss_total = train(args, data, model, optimizer, epoch)
        print("train time:{}".format(time.time() - t))
        print("Train:{}, loss:{}".format(epoch, loss_total))
        if args.tb_log and writer is not None:
            writer.add_scalar('Train/Loss', loss_total, epoch)

    return best_epoch, best_test, best_map


def parsers_parser():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=parent_path + "/data/Formatted_dataset/", help="data path")
    parser.add_argument('--dataset_str', type=str, default='Music', help='video, Beauty, Clothing, Music, Musical')
    parser.add_argument('--user_feat_type', type=str, default='smooth', help='count, smooth, smooth_imputed')
    parser.add_argument('--seed', type=int, default=0, help="Seed (For reproducability)")
    parser.add_argument('--emb_dim', type=int, default=32, help="Hidden Dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden Dimension")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")  # 0.0001

    # Training
    parser.add_argument("--feat_selection_num", type=int, default=100, help="recommending how many keep at processing")
    parser.add_argument("--out_loop", type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200, help="Number of epoch during training")
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size in one iteration")
    parser.add_argument('--test_interval', type=int, default=2)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='CNR', help='[NCF, VBPR, CER; NAR, CAR, CNR]')
    parser.add_argument("--load_pretrain", type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--tb_log', type=int, default=0)
    parser.add_argument('--txt_log', type=int, default=0)
    parser.add_argument('--norm_feat', type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_output_layer", type=int, default=2)
    parser.add_argument('--cat_feat', type=int, default=1)
    parser.add_argument('--concat_dropout', type=int, default=1)

    # CAR args
    parser.add_argument('--reg_adv', type=float, default=1, help='Regularization for adversarial loss')
    parser.add_argument('--adv_epoch', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon for adversarial weights.')
    parser.add_argument('--reg', type=float, default=0, help="Regularization for user and item embeddings.")

    # CNR args
    parser.add_argument("--confidence", type=float, default=0.65, help="should small than -0.6931471805599453")
    parser.add_argument("--intervener_feature_number", type=int, default=100, help="recommending how many items to a user")
    parser.add_argument("--intervener_iteration", type=int, default=200, help="number of training epochs")
    parser.add_argument("--intv_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--intervener_batch_size", type=int, default=500, help="tau batch size")
    parser.add_argument("--intervener_reg", type=float, default=0.0005, help="the regular item of the MF model")
    parser.add_argument("--intervener_l1_reg", type=float, default=0.001, help="the regular item of the MF model")
    parser.add_argument("--intervener_soft", type=bool, default=True, help="the regular item of the MF model")
    parser.add_argument("--anchor_model", type=int, default=1, help="the regular item of the MF model")
    parser.add_argument("--case_model", type=int, default=-1, help="the regular item of the MF model")

    # CER args
    parser.add_argument("--rec_k",  type=int, default=5, help="length of rec list")
    parser.add_argument("--lam",  type=float, default=200, help="the hyper-param for pairwise loss")
    parser.add_argument("--gam", type=float, default=1, help="the hyper-param for L1 reg")
    parser.add_argument("--alp",  type=float, default=0.05, help="margin value for pairwise loss")
    parser.add_argument("--user_mask", action="store_false", help="whether to use the user mask.")
    parser.add_argument("--vis_lr",  type=float, default=0.01, help="learning rate in optimization")
    parser.add_argument("--step",  type=int, default=250, help="# of steps in optimization")
    parser.add_argument("--mask_thresh", type=float, default=0.1, help="threshold for choosing explanations")
    parser.add_argument("--test_num", type=int, default=-1, help="the # of users to generate explanation")
    parser.add_argument("--explain_batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


def main(args):
    data = DataLoader_New(args)
    evaluator, model, i_model = init_model(args, data)

    if args.load_pretrain:
        print("loading model...")
        model_save_path = parent_path+'/saved_model/best_{}_{}.pkl'.format(args.model_name, args.dataset_str)
        model.load_state_dict(torch.load(model_save_path))
        model = model.to(args.device)
        map, mrr, p, r, f1, hit, ndcg = test(data, model, evaluator)
        print('[Test] p:{:#.4g} & r:{:#.4g} & f1:{:#.4g} & hit:{:#.4g} '
                                    '& ndcg:{:#.4g} & mrr:{:#.4g}.'.format(p, r, f1, hit, ndcg, mrr))
        vis_func(args, data, model, i_model)

    writer = init_logger(args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    g_best_epoch, g_best_map, g_best_outter, g_best_test = 0, 0, 0, ''

    if 'CNR' in args.model_name:
        warmup(args, data, model, optimizer, evaluator)

    for outter in range(args.out_loop):
        best_epoch, best_test, best_map = run(args, data, model, optimizer, evaluator, writer,
                                              intervention=i_model, is_save_model=args.save_model)
        if best_map > g_best_map:
            g_best_test = best_test
            g_best_epoch = best_epoch
            g_best_outter = outter

    if args.txt_log:
        with open(writer, 'a+') as file:
            file.write('Best Epoch:{}-{} Performance: {}\n'.format(g_best_outter, g_best_epoch, g_best_test))


if __name__ == "__main__":
    args = parsers_parser()
    args.has_counter_result = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)