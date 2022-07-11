import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import GliomaDataset
from SSL.learning import CNN_3
from SSL.learning import LeNet
from SSL.learning import AlexNet
from SSL.learning import CNN
from SSL.utils import cross_entropy_loss, softmax_mse_loss, embedding_evaluation

import os
import warnings 
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    #result = bool(os.path.exists('/Users/mac/Desktop/SSL_CNN_MIC/GLIOMA/raw/train/2100.png'))
    #print("result:", result)

    train_dataset = GliomaDataset(args, 'train')
    print("finish")
    val_dataset = GliomaDataset(args, 'val')
    print("finish")
    test_dataset = GliomaDataset(args, 'test')
    print("finish")

    tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    #model = LeNet(n_class=3).to(device)
    #model = AlexNet(n_class=3).to(device)
    #model = CNN(n_class=3).to(device)
    model = CNN_3(n_class=3).to(device)
    model = model.float()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    model_losses = []
    val_acc_curve = []
    train_acc_curve = []
    test_acc_curve = []

    for epoch in range(1, args.epochs + 1):
        tr_loss_all = 0
        val_loss_all = 0
        for batch in tr_loader:
            # set up
            #print("len(batch):", len(batch))
            model_optimizer.step()
            model.train()
            model.zero_grad()
            #classification
            img_mask = (batch[1]+batch[3]).to(device)
            #print("img_mask:", img_mask.shape)
            output = model(img_mask)
            label = batch[4].squeeze(-1).to(device)
            cls_loss = cross_entropy_loss(output, label)
            #consistency
            aug1 = model(batch[0].to(device))
            aug2 = model(batch[2].to(device))
            coss_loss = softmax_mse_loss(aug1, aug2)
            loss = cls_loss + 0.01*coss_loss
            tr_loss_all += loss.item()

            loss.backward()
            model_optimizer.step()
        fin_tr_loss = tr_loss_all / len(tr_loader)

        # 验证集的损失函数
        model.eval()
        for batch in val_loader:
            img_mask = (batch[1]+batch[3]).to(device)
            output = model(img_mask)
            label = batch[4].squeeze(-1).to(device)
            val_loss = cross_entropy_loss(output, label)
            val_loss_all += val_loss.item()
        fin_val_all = val_loss_all / len(val_loader)

        logging.info('Epoch {}, Model Loss {}, val Loss {}'.format(epoch, fin_tr_loss, fin_val_all))
        model_losses.append(fin_tr_loss)

        if epoch % args.eval_interval == 0:
            model.eval()

            train_score = embedding_evaluation(model, tr_loader, device)
            val_score = embedding_evaluation(model, val_loader, device)
            test_score = embedding_evaluation(model, test_loader, device)

            logging.info(
                "Train: acc: {} f1: {} sen: {} spe: {}".format(train_score[0], train_score[1], train_score[2], train_score[3]))

            logging.info(
                "val: acc: {} f1: {} sen: {} spe: {}".format(val_score[0], val_score[1], val_score[2], val_score[3]))

            logging.info(
                "test: acc: {} f1: {} sen: {} spe: {}".format(test_score[0], test_score[1], test_score[2], test_score[3]))

            train_acc_curve.append(train_score[0])
            val_acc_curve.append(val_score[0])
            test_acc_curve.append(test_score[0])

    model.eval()
    test_score = embedding_evaluation(model, test_loader, device)

    best_val_epoch = np.argmax(np.array(val_acc_curve))
    best_test_epoch = np.argmax(np.array(test_acc_curve))
    best_train = max(train_acc_curve)
    best_val = max(val_acc_curve)
    best_test = max(test_acc_curve)

    logging.info('FinishedTraining!')
    logging.info('BestvalEpoch: {}'.format(best_val_epoch))
    logging.info('BesttestEpoch: {}'.format(best_test_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(best_val))
    logging.info('BesttestScore: {}'.format(best_test))
    logging.info(
        "final test score: acc: {} f1: {} sen: {} spe: {}".format(test_score[0], test_score[1], test_score[2], test_score[3]))

def arg_parse():
    parser = argparse.ArgumentParser(description='SSL-CNN GLIOMA')

    parser.add_argument('--root', type=str, default='/home/zhang_istbi/zhangsj/SSL_CNN_MIC/GLIOMA',
                        help='Dataset path')
    parser.add_argument('--num', type=int, default=42,
                        help='num of data')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--batch_size', type=int, default=7,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Train Epochs')
    ############################
    parser.add_argument('--norm_factor', type=float, default=0.8, help='norm factor in dataset processing')
    parser.add_argument('--crop_size', type=float, default=0.0, help='crop_size in dataset processing')
    parser.add_argument('--eval_interval', type=int, default=1, help="eval epochs interval")
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)