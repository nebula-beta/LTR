import os
from typing import Tuple, Optional
import yaml
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Utils import init_logger, get_logger, rating_to_label_gains, rating_to_label
from DatasetAnalysis import FeatureInfo, FeatureInfo_repr, FeatureInfo_cons

logger = get_logger()

PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1

def _pad(sample, max_slate_length):
    """
    Zero padding a slate shorter than max_slate_length
    :param sample: ndarrays tuple contain features, labels of shape [slate_length, features_dim], [slate_length]
    :param max_slate_length: max state length in sample
    :return: ndarrays tuple contain features, labels and  original ranks of shape [max_slate_length, feature_dim], [max_slate_length], [max_slate_length]
    """
    x1, x2, y = sample
    slate_length = len(y)

    fixed_len_x1 = np.pad(sample[0], ((0, max_slate_length  - slate_length), (0, 0)), "constant")
    fixed_len_x2 = np.pad(sample[1], ((0, max_slate_length  - slate_length), (0, 0)), "constant")
    fixed_len_y = np.pad(sample[2], (0, max_slate_length - slate_length), "constant", constant_values=PADDED_Y_VALUE)
    indices = np.pad(np.arange(0, slate_length), (0, max_slate_length - slate_length), "constant", constant_values=PADDED_INDEX_VALUE)
    return fixed_len_x1, fixed_len_x2, fixed_len_y, indices

class LibSVMDataset(Dataset):
    """
    """
    def __init__(self, dense_feature, sparse_feature, y, query_ids, dense_feature_list, sparse_feature_list, filter_queries=True):
        """
        :param X: [dataset_size, features_dim]
        :param y: [dataset_size]
        :param query_ids: [dataset_size]
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        """

        self.num_dense_features = len(dense_feature_list)
        self.num_sparse_features= len(sparse_feature_list)
        print("self.num_dense_features", self.num_dense_features)
        print("self.num_sparse_features", self.num_sparse_features)

        self.dense_feature = dense_feature
        self.sparse_feature = sparse_feature

        unique_query_ids, indices, counts = np.unique(query_ids, return_index=True, return_counts=True)
        #indices not sorted

        groups = np.cumsum(counts[np.argsort(indices)])

        self.dense_by_qid = np.split(dense_feature, groups)[:-1] #[-1] is empty
        self.sparse_by_qid = np.split(sparse_feature, groups)[:-1] #[-1] is empty
        self.y_by_qid = np.split(y, groups)[:-1] #[-1] is empty

        dense_by_qid_filtered = []
        sparse_by_qid_filtered = []
        y_by_qid_filtered = []

        filter_count = 0
        if filter_queries:
            for x1, x2,  y in zip(self.dense_by_qid, self.sparse_by_qid, self.y_by_qid):
                if len(np.unique(y)) ==1:
                    filter_count += 1
                else:
                    dense_by_qid_filtered.append(x1)
                    sparse_by_qid_filtered.append(x2)
                    y_by_qid_filtered.append(y)


            self.dense_by_qid = dense_by_qid_filtered
            self.sparse_by_qid = sparse_by_qid_filtered
            self.y_by_qid = y_by_qid_filtered

        self.longest_slate_length = max([len(a) for a in self.dense_by_qid])
        self.shortest_slate_length = min([len(a) for a in self.dense_by_qid])
        self.average_slate_length = sum([len(a) for a in self.dense_by_qid]) / len(self.dense_by_qid)


        logger.info("Loaded dataset with {} queries".format(len(self.dense_by_qid) + filter_count))
        logger.info("{} queries got filtered".format(filter_count))
        logger.info("Loaded dataset with {} queries remaining".format(len(self.dense_by_qid)))
        logger.info("Shortest slate had {} documents".format(self.shortest_slate_length))
        logger.info("Longest slate had {} documents".format(self.longest_slate_length))
        logger.info("Average slate had {} documents".format(self.average_slate_length))


    @classmethod
    def from_tsv_file(cls, tsv_file_path, query_id_name, rating_name, feature_list, filter_queries=True, \
                     feature_process_type='normal', feature_info_dict=None, sep='\t'):
        """
        :param tsv_file_path: dataset file path
        :param query_id_name: header name of query_id
        :param rating_name: header name of rating
        :param feature_list: list contain feature list
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        :return: LibSVMDataset instantiated from given file
        """

        df = pd.read_csv(tsv_file_path, sep=sep)
        query_ids = df[query_id_name]
        if rating_name not in df:
            df[rating_name] = 0
        #y = df[rating_name].apply(rating_to_label_gains)
        y = df[rating_name].apply(rating_to_label)

        X = None
        mean = None
        std = None

        if feature_process_type == 'normal':
            X = df[feature_list]
            mean = X.mean()
            std = X.std()

            for feature in feature_list:
                X[feature] = X[feature] / std[feature] - mean[feature] / std[feature]

        elif feature_process_type == 'aether':
            dense_feature_list = []
            sparse_feature_list = []
            dense_feature = pd.DataFrame()
            sparse_feature = pd.DataFrame()

            X = df[feature_list]
            for feature in feature_list:
                feature_info = feature_info_dict[feature]

                if feature_info.unique_count == 2:
                    new_feature = feature + "_Bucket"
                    sparse_feature_list.append(new_feature)
                    sparse_feature[new_feature] = X[feature]
                    logger.info("Feature: {} {}".format(feature, "Bucket Transform"))
                    continue

                if feature_info.linear_entropy <= feature_info.log_entropy:
                    new_feature = feature + "_LogLinear"
                    dense_feature_list.append(new_feature)
                    dense_feature[new_feature] = feature_info.log_slope * X[feature].apply(lambda x : math.log(x + 1.0)) + feature_info.log_bias
                    #new_features[new_feature] = feature_info.log_slope * X[feature] + feature_info.log_bias
                    logger.info("Feature: {} {}".format(feature, "Log Linear Transform"))
                else:
                    new_feature = feature + "_Linear"
                    dense_feature_list.append(new_feature)
                    dense_feature[feature] = feature_info.linear_slope * X[feature] + feature_info.linear_bias
                    logger.info("Feature: {} {}".format(feature, "Linear Transform"))

        logger.info("Loaded dataset from {} and got dense shape {}, sparse shape {}, y shape {} and query_ids shape {}".format(tsv_file_path, dense_feature.shape, sparse_feature.shape,y.shape, query_ids.shape))

        # TODO and warning: X need sorted by query_id
        return cls(dense_feature.to_numpy(), sparse_feature.to_numpy(), y.to_numpy(), query_ids.to_numpy(), dense_feature_list, sparse_feature_list, filter_queries), mean, std, feature_info_dict


    #@property
    #def shape(self) -> list:
    #   batch_dim = len(self)
        #document_dim = self.longest_slate_length
        #features_dim = self[0][0].shape[-1]

        #return [batch_dim, document_dim, features_dim]
    def __len__(self) -> int:
        return len(self.dense_by_qid)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple contain features and labels of shapes [slate_length, features_dim] and [slate_length]
        """
        x1 = self.dense_by_qid[idx]
        x2 = self.sparse_by_qid[idx]
        y = self.y_by_qid[idx]
        #print(x)

        sample = x1, x2, y

        return sample

    @staticmethod
    def collate_fn():
        """
        Returns a collate_fn that can be used to collate batches.
        """
        def _collate_fn(batch):
            max_slate_length = max([len(y) for X1,X2,y in batch])

            fixed_len_x1_list = [] 
            fixed_len_x2_list = [] 
            fixed_len_y_list = []
            indices_list = []
            for batch_index, sample in enumerate(batch):

                x1, x2, y = sample

                #print(x.shape)
                fixed_len_x1, fixed_len_x2, fixed_len_y, indices = _pad(sample, max_slate_length)


                fixed_len_x1_list.append(fixed_len_x1)
                fixed_len_x2_list.append(fixed_len_x2)
                fixed_len_y_list.append(fixed_len_y)
                indices_list.append(indices)


            x1 = np.stack(fixed_len_x1_list, axis=0)
            x2 = np.stack(fixed_len_x2_list, axis=0)
            y = np.stack(fixed_len_y_list, axis=0)
            indices  = np.stack(indices_list, axis=0)

            return torch.FloatTensor(x1), torch.LongTensor(x2), torch.FloatTensor(y), torch.LongTensor(indices)
        return _collate_fn



class LibSVMInferenceDataset(Dataset):
    """
    """
    def __init__(self, output_columns, dense_feature, sparse_feature, dense_feature_list, sparse_feature_list):
        """
        :param X: [dataset_size, features_dim]
        :param y: [dataset_size]
        :param query_ids: [dataset_size]
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        """

        self.dense_num_features = len(dense_feature_list)
        self.sparse_num_features = len(sparse_feature_list)
        self.output_columns = output_columns.values.tolist()
        self.dense_feature = dense_feature.tolist()
        self.sparse_feature = sparse_feature.tolist()


    @classmethod
    def from_tsv_file(cls, tsv_file_path, output_column_names, feature_list, feature_process_type='normal', mean=None, std=None, feature_info_dict=None, sep='\t'):
        """
        :param tsv_file_path: dataset file path
        :param query_id_name: header name of query_id
        :param rating_name: header name of rating
        :param feature_list: list contain feature list
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        :return: LibSVMDataset instantiated from given file
        """

        df = pd.read_csv(tsv_file_path, sep=sep)
        output_columns = df[output_column_names]

        X = None

        if feature_process_type == 'normal':
            X = df[feature_list]
            mean = X.mean()
            std = X.std()

            for feature in feature_list:
                X[feature] = X[feature] / std[feature] - mean[feature] / std[feature]

        elif feature_process_type == 'aether':
            dense_feature_list = []
            sparse_feature_list = []
            dense_feature = pd.DataFrame()
            sparse_feature = pd.DataFrame()

            X = df[feature_list]
            for feature in feature_list:
                feature_info = feature_info_dict[feature]

                if feature_info.unique_count == 2:
                    new_feature = feature + "_Bucket"
                    sparse_feature_list.append(new_feature)
                    sparse_feature[new_feature] = X[feature]
                    logger.info("Feature: {} {}".format(feature, "Bucket Transform"))
                    continue

                if feature_info.linear_entropy <= feature_info.log_entropy:
                    new_feature = feature + "_LogLinear"
                    dense_feature_list.append(new_feature)
                    dense_feature[new_feature] = feature_info.log_slope * X[feature].apply(lambda x : math.log(x + 1.0)) + feature_info.log_bias
                    #new_features[new_feature] = feature_info.log_slope * X[feature] + feature_info.log_bias
                    logger.info("Feature: {} {}".format(feature, "Log Linear Transform"))
                else:
                    new_feature = feature + "_Linear"
                    dense_feature_list.append(new_feature)
                    dense_feature[feature] = feature_info.linear_slope * X[feature] + feature_info.linear_bias
                    logger.info("Feature: {} {}".format(feature, "Linear Transform"))

        logger.info("Loaded dataset from {} and got dense shape {}, sparse shape {}".format(tsv_file_path, dense_feature.shape, sparse_feature.shape))

        return cls(output_columns, dense_feature.to_numpy(), sparse_feature.to_numpy(), dense_feature_list, sparse_feature_list)

    def __len__(self) -> int:
        return len(self.dense_feature)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple contain features and labels of shapes [slate_length, features_dim] and [slate_length]
        """
        x1 = self.dense_feature[idx]
        x2 = self.sparse_feature[idx]
        output_column = self.output_columns[idx]

        return output_column, x1, x2

    #@property
    #def shape(self) -> list:
    #   batch_dim = len(self)
        #document_dim = self.longest_slate_length
        #features_dim = self[0][0].shape[-1]

        #return [batch_dim, document_dim, features_dim]

    @staticmethod
    def collate_fn():
        """
        Returns a collate_fn that can be used to collate batches.
        """
        def _collate_fn(batch):
            output_column_list = []
            x1_list = []
            x2_list = []
            for batch_index, sample in enumerate(batch):

                output_column, x1, x2 = sample
                output_column_list.append(output_column)
                x1_list.append(x1)
                x2_list.append(x2)

            return output_column_list, torch.FloatTensor(x1_list), torch.LongTensor(x2_list)
        return _collate_fn



def Test1(feature_list):
    init_logger('./Log')
    logger.info("Start Loding dataset")
    train, mean, std = LibSVMDataset.from_tsv_file('./Data/TrainSmall.tsv', 'QueryID', 'Rating', feature_list)
    X,y = train[0]

    logger.info("Dataset[0] X shape {}, y shape {}".format(X.shape, y.shape))

    collate_fn = train.collate_fn()
    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)

    seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for batch in train_loader:
        x, y, idx = batch
        print(x)
        print(y)
        print(idx)
        break
    return mean, std

def Test2(feature_list, mean, std):
    init_logger('./Log')
    logger.info("Start Loding dataset")
    output_column_names = ['QueryID','SubID', 'UrlHash16B','IsMainPath']
    train = LibSVMInferenceDataset.from_tsv_file('./Data/EvalDataWithRating.tsv', output_column_names, feature_list, mean, std)
    # train = LibSVMInferenceDataset.from_tsv_file('./Data/TrainSmall.tsv', output_column_names, feature_list, mean, std)
    output_column, x = train[0]

    # logger.info("Dataset[0] X shape {}".format(x.shape))

    collate_fn = train.collate_fn()
    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)

    seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for batch in train_loader:
        output_column, x = batch

def Test3(feature_list):
    init_logger('./Log')
    logger.info("Start Loding dataset")

    yaml.add_representer(FeatureInfo, FeatureInfo_repr)
    yaml.add_constructor(u'!FeatureInfo', FeatureInfo_cons)
    feature_info_path = './Data/TrainSmall_FeatureInfo.NumBucket_100.yaml'
    with open(feature_info_path, "r") as fi:
        feature_info_dict = yaml.load(fi.read(), Loader=yaml.FullLoader)

    train, mean, std, feature_info_dict =  LibSVMDataset.from_tsv_file('./Data/TrainSmall.tsv', 'QueryID', 'Rating', feature_list, True, 'aether', feature_info_dict)
    X1, X2 ,y = train[0]

    logger.info("Dataset[0] X1 shape {}, X2 shape {}, y shape {}".format(X1.shape, X2.shape, y.shape))

    collate_fn = train.collate_fn()
    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)

    seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for batch in train_loader:
        x1, x2, y, idx = batch
        print(x1.shape)
        print(x2.shape)
        print(y.shape)
        break

    return feature_info_dict

def Test4(feature_list, feature_info_dict):
    init_logger('./Log')
    logger.info("Start Loding dataset")
    output_column_names = ['QueryID','SubID', 'UrlHash16B','IsMainPath']
    train = LibSVMInferenceDataset.from_tsv_file('./Data/EvalDataWithRating.tsv', output_column_names, feature_list, 'aether', 'None', 'None', feature_info_dict)
    # train = LibSVMInferenceDataset.from_tsv_file('./Data/TrainSmall.tsv', output_column_names, feature_list, mean, std)
    output_column, x1, x2 = train[0]

    # logger.info("Dataset[0] X shape {}".format(x.shape))

    collate_fn = train.collate_fn()
    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)

    seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for batch in train_loader:
        output_column, x1, x2 = batch
        print(x1.shape)
        print(x2.shape)

if __name__ == '__main__':
    feature_list = 'QDVectorSimilarity,TermShardTotalQueryLength,TermShardMatchedTermCount_AUTB,TermShardMatchedTermCount_Click,TermShardMatchedPostingCount_AUTB,TermShardMatchedPostingCount_Click,TermShardMaxTermCountOfPosting_AUTB,TermShardMaxTermCountOfPosting_Click,TermShardQueryTermCoverage_AUTB,TermShardQueryTermCoverage_Click,TermShardMaxIDFOfPosting_AUTB,TermShardMaxIDFOfPosting_Click,TermShardAvgIDFOfPosting_AUTB,TermShardAvgIDFOfPosting_Click,TermShardMinPostingRatio_AUTB,TermShardMinPostingRatio_Click,BinaryTF_Url_0,BinaryTF_Url_1,BinaryTF_Url_2,BinaryTF_Url_3,BinaryTF_Url_4,BinaryTF_Url_5,BinaryTF_Url_6,BinaryTF_Url_7,BinaryTF_Url_8,BinaryTF_Url_9,BinaryTF_Title_0,BinaryTF_Title_1,BinaryTF_Title_2,BinaryTF_Title_3,BinaryTF_Title_4,BinaryTF_Title_5,BinaryTF_Title_6,BinaryTF_Title_7,BinaryTF_Title_8,BinaryTF_Title_9,BinaryTF_Body_0,BinaryTF_Body_1,BinaryTF_Body_2,BinaryTF_Body_3,BinaryTF_Body_4,BinaryTF_Body_5,BinaryTF_Body_6,BinaryTF_Body_7,BinaryTF_Body_8,BinaryTF_Body_9,AnchorFlatStreamLength,AnchorTotalPhraseCount,BodyTermCount,DomainRank,StaticRank,StreamLength_Anchor,StreamLength_Body,StreamLength_Title,StreamLength_Url,TbDomainUsers,WordsInDomain,WordsInPath,WordsInTitle,LanguagePreference,LocationPreference,d,TermShardTotalQueryTermCoverage,TermShardTotalQueryTermCoverageWeighted,TermShardMatchedTermCountWeighted_AUTB,TermShardMatchedTermCountWeighted_Click'.split(',')
    # mean, std = Test1(feature_list)
    # Test2(feature_list, mean, std)
    feature_info_dict = Test3(feature_list)
    Test4(feature_list, feature_info_dict)

