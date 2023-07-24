import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn
from torch.utils.data import SequentialSampler

import argparse
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, f1_score, auc, average_precision_score, \
    confusion_matrix, classification_report
# from sklearn.utils.fixes import signature
from inspect import signature
from sklearn.model_selection import KFold

torch.manual_seed(1)  # reproducible torch:2 np:3
np.random.seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models_xlnet import clinical_xlnet_seq, clinical_xlnet_lstm_FAST
from stream_xlnet import Data_Encoder_Seq, Data_Encoder_FAST
from configuration_xlnet import XLNetConfig
from Ranger import Ranger

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# device = torch.device("cpu")

def test_meta_finetune(data_generator, model):
    y_pred = []
    HADM_ID = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (input_id, token_types, masks, label) in enumerate(data_generator):
        score, output = model(input_id.cuda(), token_types.cuda(), masks.cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    loss = loss_accumulate / count

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), y_pred, loss.item()


def main_meta_finetune(batch_size, lr, train_epoch, config_path, model_path, save_meta_finetune_path, dataFolder,
                       prediction_label):
    lr = lr
    BATCH_SIZE = batch_size
    train_epoch = train_epoch

    loss_history = []

    # initialize xlnet config
    config = XLNetConfig.from_pretrained(config_path, num_labels=1)
    print(config)
    # load pretrained model
    model_xlnet = clinical_xlnet_seq(config)

    pretrained_dict = torch.load(model_path)
    model_dict = model_xlnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_xlnet.load_state_dict(model_dict)

    model = model_xlnet

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    print('--- Starting Data Preparation ---')

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    #    df_train=df_train.cuda()
    #    df_val=df_val.cuda()
    #    df_test=df_test.cuda()

    if prediction_label == 'Mortality':
        training_set = Data_Encoder_Seq(df_train.index.values, df_train.DEATH_90.values, df_train)
        validation_set = Data_Encoder_Seq(df_val.index.values, df_val.DEATH_90.values, df_val)
        testing_set = Data_Encoder_Seq(df_test.index.values, df_test.DEATH_90.values, df_test)
    else:
        print("Please modify the label value for your own downstream prediction task.")

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # training_generator=training_generator.cuda()
    # validation_generator=validation_generator.cuda()
    # testing_generator=testing_generator.cuda()

    opt = Ranger(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(training_generator),
                                                    epochs=train_epoch)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    print('--- Finished Data Preparation ---')

    print('--- Go for Training ---')
    for epo in range(train_epoch):
        model.train()
        for i, (input_id, token_types, masks, label) in enumerate(training_generator):
            score, output = model(input_id.cuda(), token_types.cuda(), masks.cuda())
            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            #print("loss-type", loss.type)
#            print("loss_history-0", loss_history.dtype)
            loss_history.append(float(loss.cpu().detach().numpy()))

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            if (i % 200 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, logits, loss = test_meta_finetune(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
                path = save_meta_finetune_path
                # torch.save(model.module.state_dict(), path)
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), path)
                else:
                    torch.save(model.state_dict(), path)
            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc))

    print('--- Go for Testing (Validation During Tuning) ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, logits, loss = test_meta_finetune(testing_generator, model_max)
            print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , Test loss: ' + str(loss))
    except:
        print('testing failed')
    return model_max, loss_history


#### Fine-tuning, Generate the fixed embeddings and store it to save time

def generate(data_generator, model):
    y_pred = []
    y_label = []
    y_output = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (input_id, token_types, masks, label) in tqdm(enumerate(data_generator)):
        score, output = model(input_id.cuda(), token_types.cuda(), masks.cuda())
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        y_output.append(outputs)

    loss = loss_accumulate / count
    return roc_auc_score(y_label, y_pred), y_output, y_pred


def process(outputs, file_path):
    output_seq = torch.cat(outputs, 0)
    torch.save(output_seq, file_path)


def meta_finetune_embed(config_path, model_path, dataFolder, prediction_label, saving_notes_embed_batch_size):
    # initialize xlnet config
    config = XLNetConfig.from_pretrained(config_path, num_labels=1)

    # load pretrained model
    model_xlnet = clinical_xlnet_seq(config)

    pretrained_dict = torch.load(model_path)
    model_dict = model_xlnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_xlnet.load_state_dict(model_dict)

    model = model_xlnet
    model = model.cuda()

    model.eval()

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    if prediction_label == 'Mortality':
        training_set = Data_Encoder_Seq(df_train.index.values, df_train.DEATH_90.values, df_train)
        validation_set = Data_Encoder_Seq(df_val.index.values, df_val.DEATH_90.values, df_val)
        testing_set = Data_Encoder_Seq(df_test.index.values, df_test.DEATH_90.values, df_test)
    else:
        print("Please modify the label value for your own downstream prediction task.")

    training_generator = data.DataLoader(training_set, sampler=SequentialSampler(training_set),
                                         batch_size=saving_notes_embed_batch_size)
    validation_generator = data.DataLoader(validation_set, sampler=SequentialSampler(validation_set),
                                           batch_size=saving_notes_embed_batch_size)
    testing_generator = data.DataLoader(testing_set, sampler=SequentialSampler(testing_set),
                                        batch_size=saving_notes_embed_batch_size)

    with torch.set_grad_enabled(False):
        _, outputs, _ = generate(training_generator, model)
        process(outputs, dataFolder + '/train_doc_emb.pt')

    with torch.set_grad_enabled(False):
        _, outputs, _ = generate(validation_generator, model)
        process(outputs, dataFolder + '/val_doc_emb.pt')

    with torch.set_grad_enabled(False):
        _, outputs, logits = generate(testing_generator, model)
        process(outputs, dataFolder + '/test_doc_emb.pt')


def test_finetune(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (output, label) in enumerate(data_generator):
        score = model(output)

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1
        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        #print("outputs:", outputs)
    loss = loss_accumulate / count
    # print('1',logits,y_pred)
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), y_label, y_pred, outputs, loss.item()

def SCORE(label, scores, pre, loss):
    # print('3',scores,pre)
    fpr,tpr,thresholds=roc_curve(label, scores)
    # print('FPR:',fpr)
    # print('TPR:',tpr)
    # print('thresholds:',thresholds)
    AUROC = roc_auc_score(label, scores)
    plt.figure()
    plt.plot(fpr, tpr, '-', label="ROC(area={0:.2f})".format(AUROC), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/ROC.png")


    AUPRC = average_precision_score(label, scores)
    p, r, t = precision_recall_curve(label, scores)
    plt.figure()
    plt.plot(r, p, '-', label="PRC(area={0:.2f})".format(AUPRC), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/PRC.png")

    matrix = confusion_matrix(label, pre)
    TP = matrix[1, 1]
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    Accuracy = (TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    PPV = TP / float(TP + FP)
    Recall = TP / float(TP + FN)
    Precision = TP / float(TP + FP)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    True_Positive_Rate = TP / float(TP + FN)
    False_Positive_Rate = FP / float(FP + TN)

    print('Accuracy:', Accuracy)
    print('Sensitivity:', Sensitivity)
    print('Specificity:', Specificity)
    print('PPV:', PPV)
    print('Recall:', Recall)
    print('Precision:', Precision)
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    print('F1-score:', F1)
    print('True Positive Rate:', True_Positive_Rate)
    print('False Positive Rate:', False_Positive_Rate)
    print('classification:', classification_report(label, pre))
    # a = {'MODEL': ['T'],'AUROC':[AUROC],'AUPRC':[AUPRC],'LOSS':[loss],'ACCRA':[Accuracy],'Sensitivity':[Sensitivity],
    #      'Specificity':[Specificity],'PPV':[PPV],'Recall':Recall,'Precision':[Precision],'F1':[F1],'True Positive Rate':[True_Positive_Rate],'False Positive Rate':[False_Positive_Rate]}
    a = {'MODEL': ['linear'],
         'AUROC': [AUROC],
         'AUPRC': [AUPRC],
         'LOSS': [loss],
         'ACCRA':[Accuracy],
         'Sensitivity':[Sensitivity],
         'Specificity':[Specificity],
         'PPV':[PPV],
         'Recall':[Recall],
         'Precision':[Precision],
         'F1':[F1],
         'True Positive Rate':[True_Positive_Rate],
         'False Positive Rate':[False_Positive_Rate]}

    dfa = pd.DataFrame(a)

    # mode = 'a'为追加数据，index为每行的索引序号，header为标题
    print('----------writing------------')
    dfa.to_csv('/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/sum.csv', mode='a', index=False, header=False)
    print('----------finished------------')

    return fpr, tpr, thresholds, p, r, t


def main_finetune(lr, batch_size, train_epoch, dataFolder, prediction_label):
    lr = lr
    BATCH_SIZE = batch_size
    train_epoch = train_epoch
    df1 = pd.DataFrame()

    loss_history = []

    model = clinical_xlnet_lstm_FAST()
    model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    #
    print('--- Data Preparation ---')

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    doc_train = torch.load(dataFolder + '/train_doc_emb.pt')
    doc_val = torch.load(dataFolder + '/val_doc_emb.pt')
    doc_test = torch.load(dataFolder + '/test_doc_emb.pt')

    def doc_emb_to_df(doc, df):
        output_seq = [torch.unsqueeze(doc[i], 0).cpu().detach().numpy() for i in range(doc.shape[0])]
        df = df.assign(DOC_EMB=output_seq)
        return df

    df_train = doc_emb_to_df(doc_train, df_train)
    df_val = doc_emb_to_df(doc_val, df_val)
    df_test = doc_emb_to_df(doc_test, df_test)

    if prediction_label == 'Mortality':
        train_unique = df_train[['HADM_ID', 'DEATH_90']].drop_duplicates().reset_index(drop=True)
        val_unique = df_val[['HADM_ID', 'DEATH_90']].drop_duplicates().reset_index(drop=True)
        test_unique = df_test[['HADM_ID', 'DEATH_90']].drop_duplicates().reset_index(drop=True)

        training_set = Data_Encoder_FAST(train_unique.HADM_ID.values, train_unique.DEATH_90.values, df_train)
        training_generator = data.DataLoader(training_set, **params)
        #print("type(training_generator):", type(training_generator))
        validation_set = Data_Encoder_FAST(val_unique.HADM_ID.values, val_unique.DEATH_90.values, df_val)
        validation_generator = data.DataLoader(validation_set, **params)

        testing_set = Data_Encoder_FAST(test_unique.HADM_ID.values, test_unique.DEATH_90.values, df_test)
        testing_generator = data.DataLoader(testing_set, **params)
    else:
        print("Please modify the label value for your own downstream prediction task.")

    # print("training_generator:",training_generator[0:4])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(train_epoch):
        model.train()
        for i, (output, label) in enumerate(training_generator):
            score = model(output.cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            # loss_history.append(loss)
            loss_history.append(loss.cpu().detach().numpy())


            opt.zero_grad()
            loss.backward()

            opt.step()

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, ylabel, logits, ypre, loss = test_finetune(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc

            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc))

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, ylabel, logits, ypre, loss = test_finetune(testing_generator, model_max)
            print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , Test loss: ' + str(loss))
            # print('F1:', f1_score(ylabel, ypre))
            # print('classi:', classification_report(ylabel, ypre))
            # print('1')
            # print('2',logits,ypre)
            fpr, tpr, thresholds, p, r, t = SCORE(ylabel, logits, ypre, loss)
            # print('2:',fpr,tpr,thresholds, p, r, t)
            df1 = {'label': ylabel,
                 'scores': logits,
                 'pre': ypre,
                 'loss': loss_history,
                 'fpr': fpr,
                 'tpr': tpr,
                 't1': thresholds,
                 'precision': p,
                 'recall': r,
                 't2': t}

            pd.DataFrame(pd.DataFrame.from_dict(df1, orient='index').values.T, columns=list(df1.keys())).to_csv(
                '/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/linear.csv', index=False, mode='w')

            # print('3')
    except Exception as e:
        print('testing failed')
        print(e.args)
        print(str(e))
        print(repr(e))

    # df1.to_csv("/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/TGRA_1.csv")
    return model_max, loss_history

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data folder. Should contain the train/val/test csv files for the task.")
    parser.add_argument("--config_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Clinical XLNet configuration path.")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Clinical XLNet pretrained model path.")
    parser.add_argument("--save_meta_finetune_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Clinical XLNet meta finetuned model path.")
    parser.add_argument("--prediction_label",
                        default=None,
                        type=str,
                        required=True,
                        help="Mortality")
    parser.add_argument("--Batch_Size_Meta",
                        default=32,
                        type=int,
                        help="Batch size for meta finetuned stage")
    parser.add_argument("--Learning_Rate_Meta",
                        default=1e-5,
                        type=float,
                        help="Learning rate for meta finetuned stage")
    parser.add_argument("--Training_Epoch_Meta",
                        default=4,
                        type=int,
                        help="Training Epoch Numbers for meta finetuned stage")
    parser.add_argument("--Batch_Size_Finetune",
                        default=128,
                        type=int,
                        help="Batch size for Finetune stage")
    parser.add_argument("--Learning_Rate_Finetune",
                        default=2e-5,
                        type=float,
                        help="Learning rate for Finetune stage")
    parser.add_argument("--Training_Epoch_Finetune",
                        default=20,
                        type=int,
                        help="Training Epoch Numbers for Finetune stage")
    parser.add_argument("--saving_notes_embed_batch_size",
                        default=64,
                        type=int,
                        help="Saving Notes Embedding Batch Size to facilitate the Finetuning stage")
    parser.add_argument("--skip_meta_finetuned",
                        default=True,
                        action='store_true',
                        help="If you want to load the model path provided for Death Prediction.")
    parser.add_argument("--skip_note_embed_save",
                        default=True,
                        action='store_true',
                        help="If you already save the note embedding.")

    args = parser.parse_args()

    config_path = args.config_path
    model_path = args.model_path
    save_meta_finetune_path = args.save_meta_finetune_path
    dataFolder = args.data_dir

    if not args.skip_meta_finetuned:

        print("----- meta finetuning stage -----")
        # meta-finetune
        s = time()
        model_max, loss_history = main_meta_finetune(args.Batch_Size_Meta, args.Learning_Rate_Meta,
                                                     args.Training_Epoch_Meta, config_path, model_path,
                                                     save_meta_finetune_path, dataFolder, args.prediction_label)
        e = time()
        print(e - s)
        plt.figure()
        plt.plot(loss_history)
        plt.show()
        plt.savefig("/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/meta_finetuning_stage.png")

    else:
        print("------ meta finetuning skipped, directly loading pretrained model from save_meta_finetune_path -----")

    if not args.skip_note_embed_save:
        print("----- saving note embedding to facilitate the finetuning stage -----")
        # save note embedding
        meta_finetune_embed(config_path, save_meta_finetune_path, dataFolder, args.prediction_label,
                            args.saving_notes_embed_batch_size)
    else:
        print(
            "----- note embedding is already saved in the desigated file folder, if not, delete skip_note_embed_save line -----")

    print("----- finetuning stage -----")
    # finetune
    s = time()
    model_max, loss_history = main_finetune(args.Learning_Rate_Finetune, args.Batch_Size_Finetune,args.Training_Epoch_Finetune, dataFolder, args.prediction_label)
    # string = 'TGRA_1.csv'
    e = time()
    print(e - s)
    plt.plot(loss_history)
    plt.show()
    plt.savefig("/root/userfolder/clinicalXLNet-master/clinicalXLNet-master/save/finetuning_stage.png")

if __name__ == "__main__":
    main()
