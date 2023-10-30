from typing import Optional
import argparse
import os
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


from Utils import init_logger, get_logger, send_to_bark
from Dataset import LibSVMDataset, LibSVMInferenceDataset
from Model import NeuralNet, FM_model, FM_model2
from Loss import approxNDCGLoss, lambdaLoss, listMLE, ordinal, rankNet
from Metrics import Metrics
from DatasetAnalysis import FeatureInfo, FeatureInfo_repr, FeatureInfo_cons

def training(args):

    init_logger(args.log_dir, args.exp_name + ".log")
    logger = get_logger()
    writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))

    logger.info("args : {}".format(args))


    seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    feature_list = 'QDVectorSimilarity,TermShardTotalQueryLength,TermShardMatchedTermCount_AUTB,TermShardMatchedTermCount_Click,TermShardMatchedPostingCount_AUTB,TermShardMatchedPostingCount_Click,TermShardMaxTermCountOfPosting_AUTB,TermShardMaxTermCountOfPosting_Click,TermShardQueryTermCoverage_AUTB,TermShardQueryTermCoverage_Click,TermShardMaxIDFOfPosting_AUTB,TermShardMaxIDFOfPosting_Click,TermShardAvgIDFOfPosting_AUTB,TermShardAvgIDFOfPosting_Click,TermShardMinPostingRatio_AUTB,TermShardMinPostingRatio_Click,BinaryTF_Url_0,BinaryTF_Url_1,BinaryTF_Url_2,BinaryTF_Url_3,BinaryTF_Url_4,BinaryTF_Url_5,BinaryTF_Url_6,BinaryTF_Url_7,BinaryTF_Url_8,BinaryTF_Url_9,BinaryTF_Title_0,BinaryTF_Title_1,BinaryTF_Title_2,BinaryTF_Title_3,BinaryTF_Title_4,BinaryTF_Title_5,BinaryTF_Title_6,BinaryTF_Title_7,BinaryTF_Title_8,BinaryTF_Title_9,BinaryTF_Body_0,BinaryTF_Body_1,BinaryTF_Body_2,BinaryTF_Body_3,BinaryTF_Body_4,BinaryTF_Body_5,BinaryTF_Body_6,BinaryTF_Body_7,BinaryTF_Body_8,BinaryTF_Body_9,AnchorFlatStreamLength,AnchorTotalPhraseCount,BodyTermCount,DomainRank,StaticRank,StreamLength_Anchor,StreamLength_Body,StreamLength_Title,StreamLength_Url,TbDomainUsers,WordsInDomain,WordsInPath,WordsInTitle,LanguagePreference,LocationPreference,d,TermShardTotalQueryTermCoverage,TermShardTotalQueryTermCoverageWeighted,TermShardMatchedTermCountWeighted_AUTB,TermShardMatchedTermCountWeighted_Click'.split(',')
    logger.info("Start Loding dataset")
    output_column_names = ['QueryID','SubID', 'UrlHash16B','IsMainPath']
    if args.feature_info_path != '':
        yaml.add_representer(FeatureInfo, FeatureInfo_repr)
        yaml.add_constructor(u'!FeatureInfo', FeatureInfo_cons)
        with open(args.feature_info_path, "r") as fi:
            feature_info_dict = yaml.load(fi.read(), Loader=yaml.FullLoader)

        train_data, mean, std, feature_info_dict = LibSVMDataset.from_tsv_file(args.train_data, 'QueryID', 'Rating', feature_list, True, 'aether', feature_info_dict)
        test_data = LibSVMInferenceDataset.from_tsv_file(args.test_data, output_column_names, feature_list, 'aether', None, None, feature_info_dict)
    else:
        train_data, mean, std  = LibSVMDataset.from_tsv_file(args.train_data, 'QueryID', 'Rating', feature_list, filter_queries=True)
        test_data = LibSVMInferenceDataset.from_tsv_file(args.test_data, output_column_names, feature_list, mean, std)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100000, shuffle=False, collate_fn=test_data.collate_fn())

    # with open("mean_std.tsv", "w") as fo:
    #     for feature in feature_list:
    #         fo.write("{}\t{}\t{}\n".format(feature, mean[feature], std[feature]))

    list(map(int, args.hidden_nodes.split(',')))
    # Setup model, optimizer and loss
    #model = NeuralNet(train_data.num_features, 4, [64,32,15])
    #model = NeuralNet(train_data.num_features, 3, [15, 6])
    #model = NeuralNet(train_data.num_features, args.layer, list(map(int, args.hidden_nodes.split(','))))
    model = NeuralNet(train_data.num_features, args.layer, list(map(int, args.hidden_nodes.split(','))))
    #model = FM_model2(train_data.num_features, 5)
    # model = FM_model(train_data.num_features, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e:(3000-e)**2/100, last_epoch=-1)
    # torch.optim.lr_scheduler.ReduceLROnPlateau

    if args.finetune is not None:
        model.load_state_dict(torch.load(args.ckpt))
        pass

    idealset = pd.read_csv('./Data/EvalIdealset.tsv', sep='\t')

    logger.info("Start Training")
    best_fidelity = 0
    best_ndcg = 0
    best_epoch = 0
    for epoch in range(args.epochs):

        logger.info("Training")
        model.train()
        for batch in train_loader:
            x, y, idx = batch


            pred = model(x)
            #weight = (1 - y / 100)
            #pred = pred * weight[:, :, None]

            pred = pred.reshape(pred.shape[0], pred.shape[1])
            # loss = approxNDCGLoss(pred, y, alpha=args.alpha)
            # loss = rankNet(pred, y, weight_by_diff=True)

            loss = lambdaLoss(pred, y, weighing_scheme='ndcgLoss1_scheme', sigma=1.0, reduction_log='natural', reduction='sum')
            # loss = lambdaLoss(pred, y, weighing_scheme='ndcgLoss2_scheme', sigma=1.0)
            # loss = lambdaLoss(pred, y, weighing_scheme='ndcgLoss2PP_scheme', sigma=1.0)
            # loss = listMLE(pred, y)
            #loss = ordinal(pred, y, 5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #scheduler.step()
        if epoch != 0 and epoch % 1 == 0:
            logger.info("Eval")
            model.eval()
            output_column_list = []
            predict_list = []
            with torch.no_grad():
                for batch in test_loader:
                    output_column, x = batch

                    x = x.reshape(-1, x.shape[-1])
                    predict_data = model(x).detach().numpy().tolist()


                    output_column_list.extend(output_column)
                    predict_list.extend(predict_data)


            logger.info("Calc")
            recallset = pd.DataFrame(output_column_list, columns=output_column_names)
            recallset['L1Rank'] = pd.DataFrame(predict_list, columns=['L1Rank'])
            fidelity = Metrics.calc_fidelity(idealset, recallset)



            # logger.info("Epoch: {}, Ndcg: {}, Fidelity: {}, Fidelity2 : {}".format(epoch, ndcg, fidelity, fidelity2))
            logger.info("Epoch: {},  Fidelity: {}".format(epoch, fidelity))
            writer.add_scalar('Fidelity', fidelity, epoch)


        #save the model
        # if ndcg > best_ndcg:
        #     best_ndcg = ndcg
        #     best_epoch = epoch
        #     if not os.path.exists(args.model_dir):
        #         os.mkdir(args.model_dir)
        #     torch.save(model.state_dict(), os.path.join(args.model_dir, args.exp_name + '.pkl'),  _use_new_zipfile_serialization=False)
    # logger.info("Best Ndcg : {}, at Epoch: {}".format(best_ndcg, best_epoch))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train neural network based on data and the ini file")
    parser.add_argument('--train_data', dest = "train_data", type = str, required = True, help = "input train data tsv file")
    parser.add_argument('--test_data', dest = "test_data", type = str, required = True, help = "input test data tsv file")
    parser.add_argument('--feature_info_path', dest = "feature_info_path", type = str, default='', required = False, help = "input test data tsv file")
    parser.add_argument('--log_dir', dest = "log_dir", type = str, default = './Log', help = "log dir")
    parser.add_argument('--model_dir', dest = "model_dir", type = str, default = './Model', help = "log dir")
    parser.add_argument('--exp_name', dest = "exp_name", type = str, required = True, help = "experiment name")
    parser.add_argument('--epochs', dest = "epochs", type = int, default = 150, help = "max epochs allowed (default = 150)")
    parser.add_argument('--lr', dest = "lr", type = float, default = 1e-3, help = "initial learning rate (default = 1e-3)")
    parser.add_argument('--batch_size', dest = "batch_size", type = int, default = 32, help = "Training batch size. (default = 32)")
    parser.add_argument('--alpha', dest = "alpha", type = float, default = 1.0, help = "alpha")
    parser.add_argument('--layer', dest = "layer", type = int, default = 2, help = "Neural network layer. (default = 3)")
    parser.add_argument('--hidden_nodes', dest = "hidden_nodes", type = str, default = "15", help = "Neural network hidden nodes, sep by ,(default=15,6)")
    parser.add_argument('--finetune', dest = "finetune", type = str, default = None, help = "is finetune or not")
    parser.add_argument('--ckpt', dest = "ckpt", type = str, default =None, help = "ckpt path")


    args = parser.parse_args()


    title = "Exp : {}".format(args.exp_name)
    content = "args : {}".format(args)
    #send_to_bark(title, "Start")

    training(args)

    #send_to_bark(title, "End")
