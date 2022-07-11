import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import GliomaDataset
from SSL.learning import GoogLeNet
from SSL.learning import CNN_4
from SSL.learning import AlexNet
from SSL.learning import CNN
from SSL.utils import cross_entropy_loss, softmax_mse_loss, embedding_evaluation

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import os
import warnings 
warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    #train_dataset = GliomaDataset(args, 'train')
    #print("finish")
    #val_dataset = GliomaDataset(args, 'val')
    #print("finish")
    test_dataset = GliomaDataset(args, 'test')
    print("finish")
    
    dataset = GliomaDataset(args, 'train_val')
    #id = [i+1 for i in range(35)]
    kfold = 5
    kf = KFold(n_splits=kfold, random_state=123, shuffle = True)

    #tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    #model = GoogLeNet(drop_ratio=args.drop_ratio, n_class=3).to(device)
    #model = CNN_4(n_class=3).to(device)
    #model = AlexNet(n_class=3).to(device)
    #model = CNN(n_class=3).to(device)
    #model = model.float()
    #model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    model_losses = []
    test_acc_curve = []
    test_f1_curve = []
    test_sen_curve = []
    test_spe_curve = []
    val_acc_curve = []
    train_acc_curve = []
    acc_best_test = []
    f1_best_test = []
    sen_best_test = []
    spe_best_test = []

    acc = []
    f1 = [] 
    sen = [] 
    spe = []

    for fold, (tr_id, val_id) in enumerate(kf.split(dataset)):
        #model = CNN_4(n_class=3).to(device)
        # CNN_4是一个四卷积模型
        model = CNN_4(channel=2, n_class=2).to(device)
        model = model.float()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
            
        train_sampler = SubsetRandomSampler(tr_id)
        valid_sampler = SubsetRandomSampler(val_id)
        tr_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

        for epoch in range(1, args.epochs + 1):

            tr_loss_all = 0
            for batch in tr_loader:
                #print("batch[0].shape:", batch[0].shape)
                # [:,:2,:,:]QSM_T1, [:,1:,:,:]T1_T2,concate([:,0,:,:],[:,2,:,:],dim=0)，QSM_T2
                # [:,2,:,:]=t2,[:,1,:,:]=t1,[:,0,:,:]=qsm
                model_optimizer.step()
                model.train()
                model.zero_grad()
                #classification
                img = torch.concat((batch[1][:,0,:,:].unsqueeze(dim=1),batch[1][:,2,:,:].unsqueeze(dim=1)), dim=1) 
                mask = torch.concat((batch[3][:,0,:,:].unsqueeze(dim=1),batch[3][:,2,:,:].unsqueeze(dim=1)), dim=1) 
                img_mask = (img+mask).to(device)
                #img_mask = img_mask.unsqueeze(dim=1).to(device)
                #print("img_mask:", img_mask.shape)
                output = model(img_mask)
                label = batch[4].squeeze(-1).to(device)
                cls_loss = cross_entropy_loss(output, label)
                #consistency
                aug1 = model(torch.concat((batch[0][:,0,:,:].unsqueeze(dim=1),batch[0][:,2,:,:].unsqueeze(dim=1)), dim=1).to(device))
                aug2 = model(torch.concat((batch[2][:,0,:,:].unsqueeze(dim=1),batch[2][:,2,:,:].unsqueeze(dim=1)), dim=1).to(device))
                coss_loss = softmax_mse_loss(aug1, aug2)
                loss = cls_loss + 0.5*coss_loss
                tr_loss_all += loss.item()

                loss.backward()
                model_optimizer.step()
            fin_tr_loss = tr_loss_all / len(tr_loader)

            # 验证集的损失函数
            model.eval()
            val_loss_all = 0
            for batch in val_loader:
                img = torch.concat((batch[1][:,0,:,:].unsqueeze(dim=1),batch[1][:,2,:,:].unsqueeze(dim=1)), dim=1) 
                mask = torch.concat((batch[3][:,0,:,:].unsqueeze(dim=1),batch[3][:,2,:,:].unsqueeze(dim=1)), dim=1) 
                img_mask = (img+mask).to(device)
                #img_mask = (batch[1][:,:2,:,:]+batch[3][:,:2,:,:]).to(device)
                output = model(img_mask)
                label = batch[4].squeeze(-1).to(device)
                val_loss = cross_entropy_loss(output, label)
                val_loss_all += val_loss.item()
            fin_val_all = val_loss_all / len(val_loader)

            logging.info('Epoch {}, Model Loss {}, val Loss {}'.format(epoch, fin_tr_loss, fin_val_all))
            model_losses.append(fin_tr_loss)

            train_score = embedding_evaluation(model, tr_loader, device)
            val_score = embedding_evaluation(model, val_loader, device)
            test_score = embedding_evaluation(model, test_loader, device)

            if epoch % args.eval_interval == 0:
                model.eval()
                logging.info(
                    "Train: acc: {} f1: {} sen: {} spe: {}".format(train_score[0], train_score[1], train_score[2], train_score[3]))

                logging.info(
                    "val: acc: {} f1: {} sen: {} spe: {}".format(val_score[0], val_score[1], val_score[2], val_score[3]))

                train_acc_curve.append(train_score[0])
                val_acc_curve.append(val_score[0])
                test_acc_curve.append(test_score[0])
                test_f1_curve.append(test_score[1])
                test_sen_curve.append(test_score[2])
                test_spe_curve.append(test_score[3])
        
        acc_best_test.append(max(test_acc_curve))
        f1_best_test.append(max(test_f1_curve))
        sen_best_test.append(max(test_sen_curve))
        spe_best_test.append(max(test_spe_curve))

        model.eval()
        test_acc, test_f1, test_sen, test_spe = embedding_evaluation(model, test_loader, device)

        logging.info(
            "fold: {} acc: {} f1: {} sen: {} spe: {}".format(fold, test_acc, test_f1, test_sen, test_spe))

        acc.append(test_acc)
        f1.append(test_f1)
        sen.append(test_sen)
        spe.append(test_spe)

    best_val_epoch = np.argmax(np.array(val_acc_curve))
    best_train = max(train_acc_curve)
    best_val = max(val_acc_curve)

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(best_val))
    logging.info(
        "best_test_mean: acc: {} f1: {} sen: {} spe: {}".format(np.array(acc_best_test).mean(), np.array(f1_best_test).mean(), np.array(sen_best_test).mean(), np.array(spe_best_test).mean()))
    logging.info(
        "best_test_std: acc: {} f1: {} sen: {} spe: {}".format(np.array(acc_best_test).std(), np.array(f1_best_test).std(), np.array(sen_best_test).std(), np.array(spe_best_test).std()))
    logging.info(
        "test_mean: acc: {} f1: {} sen: {} spe: {}".format(np.array(acc).mean(), np.array(f1).mean(), np.array(sen).mean(), np.array(spe).mean()))
    logging.info(
        "test_std: acc: {} f1: {} sen: {} spe: {}".format(np.array(acc).std(), np.array(f1).std(), np.array(sen).std(), np.array(spe).std()))

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
    parser.add_argument('--eval_interval', type=int, default=1, help="eval epochs interval")
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)