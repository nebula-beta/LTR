import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Utils import init_logger, get_logger, rating_to_label_gains, rating_to_label

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
    x, y = sample
    slate_length = len(y)

    fixed_len_x = np.pad(sample[0], ((0, max_slate_length  - slate_length), (0, 0)), "constant")
    fixed_len_y = np.pad(sample[1], (0, max_slate_length - slate_length), "constant", constant_values=PADDED_Y_VALUE)
    indices = np.pad(np.arange(0, slate_length), (0, max_slate_length - slate_length), "constant", constant_values=PADDED_INDEX_VALUE)
    return fixed_len_x, fixed_len_y, indices

class LibSVMDataset(Dataset):
    """
    """
    def __init__(self, X, y, query_ids, feature_list, filter_queries=True):
        """
        :param X: [dataset_size, features_dim]
        :param y: [dataset_size]
        :param query_ids: [dataset_size]
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        """

        self.num_features = len(feature_list)

        #self.mean = np.mean(X, axis=0) 
        #self.std = np.std(X, axis=0, ddof=1)
        #self.X = X / self.std - self.mean / self.std
        self.X = X

        unique_query_ids, indices, counts = np.unique(query_ids, return_index=True, return_counts=True)
        #indices not sorted

        groups = np.cumsum(counts[np.argsort(indices)])

        self.X_by_qid = np.split(X, groups)[:-1] #[-1] is empty
        self.y_by_qid = np.split(y, groups)[:-1] #[-1] is empty

        X_by_qid_filtered = []
        y_by_qid_filtered = []

        filter_count = 0
        if filter_queries:
            for x, y in zip(self.X_by_qid, self.y_by_qid):
                if len(np.unique(y)) ==1:
                    filter_count += 1
                else:
                    X_by_qid_filtered.append(x)
                    y_by_qid_filtered.append(y)


            self.X_by_qid = X_by_qid_filtered
            self.y_by_qid = y_by_qid_filtered

        self.longest_slate_length = max([len(a) for a in self.X_by_qid])
        self.shortest_slate_length = min([len(a) for a in self.X_by_qid])
        self.average_slate_length = sum([len(a) for a in self.X_by_qid]) / len(self.X_by_qid)


        logger.info("Loaded dataset with {} queries".format(len(self.X_by_qid) + filter_count))
        logger.info("{} queries got filtered".format(filter_count))
        logger.info("Loaded dataset with {} queries remaining".format(len(self.X_by_qid)))
        logger.info("Shortest slate had {} documents".format(self.shortest_slate_length))
        logger.info("Longest slate had {} documents".format(self.longest_slate_length))
        logger.info("Average slate had {} documents".format(self.average_slate_length))


    @classmethod
    def from_tsv_file(cls, tsv_file_path, query_id_name, rating_name, feature_list, filter_queries=True, transform=False, mean=None, std=None, sep='\t'):
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
        X = df[feature_list]

        if not transform:
            mean = df[feature_list].mean()
            std = df[feature_list].std()


        for feature in feature_list:
            X[feature] = X[feature] / std[feature] - mean[feature] / std[feature]


        logger.info("Loaded dataset from {} and got X shape {}, y shape {} and query_ids shape {}".format(tsv_file_path, X.shape, y.shape, query_ids.shape))

        # TODO and warning: X need sorted by query_id
        return cls(X.to_numpy(), y.to_numpy(), query_ids.to_numpy(), feature_list, filter_queries), mean, std


    #@property
    #def shape(self) -> list:
    #   batch_dim = len(self)
        #document_dim = self.longest_slate_length
        #features_dim = self[0][0].shape[-1]

        #return [batch_dim, document_dim, features_dim]
    def __len__(self) -> int:
        return len(self.X_by_qid)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple contain features and labels of shapes [slate_length, features_dim] and [slate_length]
        """
        x = self.X_by_qid[idx]
        y = self.y_by_qid[idx]
        #print(x)

        sample = x, y

        return sample

    @staticmethod
    def collate_fn():
        """
        Returns a collate_fn that can be used to collate batches.
        """
        def _collate_fn(batch):
            max_slate_length = max([len(y) for X,y in batch])

            fixed_len_x_list = [] 
            fixed_len_y_list = []
            indices_list = []
            for batch_index, sample in enumerate(batch):

                x, y = sample

                #print(x.shape)
                fixed_len_x, fixed_len_y, indices = _pad(sample, max_slate_length)


                fixed_len_x_list.append(fixed_len_x)
                fixed_len_y_list.append(fixed_len_y)
                indices_list.append(indices)

                #print(fixed_len_x.shape)
                #print(fixed_len_y.shape)
                #print(indices.shape)

            x = np.stack(fixed_len_x_list, axis=0)
            y = np.stack(fixed_len_y_list, axis=0)
            indices  = np.stack(indices_list, axis=0)

            return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor(indices)
        return _collate_fn



class LibSVMInferenceDataset(Dataset):
    """
    """
    def __init__(self, output_columns, X, feature_list):
        """
        :param X: [dataset_size, features_dim]
        :param y: [dataset_size]
        :param query_ids: [dataset_size]
        :param filter_queries: whether to filter queries that have no relevant documents associated with them.
        """

        self.num_features = len(feature_list)
        self.output_columns = output_columns.values.tolist()
        self.X = X.tolist()


    @classmethod
    def from_tsv_file(cls, tsv_file_path, output_column_names, feature_list, mean, std, sep='\t'):
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
        X = df[feature_list]

        for feature in feature_list:
            X[feature] = X[feature] / std[feature] - mean[feature] / std[feature]


        logger.info("Loaded dataset from {} and got X shape {}".format(tsv_file_path, X.shape))

        return cls(output_columns, X.to_numpy(), feature_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple contain features and labels of shapes [slate_length, features_dim] and [slate_length]
        """
        x = self.X[idx]
        output_column = self.output_columns[idx]

        return output_column, x

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
            x_list = []
            for batch_index, sample in enumerate(batch):

                output_column, x = sample
                output_column_list.append(output_column)
                x_list.append(x)

            return output_column_list, torch.FloatTensor(x_list)
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
    train = LibSVMInferenceDataset.from_tsv_file('./Data/Eval.tsv', output_column_names, feature_list, mean, std)
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


if __name__ == '__main__':
    feature_list = 'QDVectorSimilarity,TermShardTotalQueryLength,TermShardMatchedTermCount_AUTB,TermShardMatchedTermCount_Click,TermShardMatchedPostingCount_AUTB,TermShardMatchedPostingCount_Click,TermShardMaxTermCountOfPosting_AUTB,TermShardMaxTermCountOfPosting_Click,TermShardQueryTermCoverage_AUTB,TermShardQueryTermCoverage_Click,TermShardMaxIDFOfPosting_AUTB,TermShardMaxIDFOfPosting_Click,TermShardAvgIDFOfPosting_AUTB,TermShardAvgIDFOfPosting_Click,TermShardMinPostingRatio_AUTB,TermShardMinPostingRatio_Click,BinaryTF_Url_0,BinaryTF_Url_1,BinaryTF_Url_2,BinaryTF_Url_3,BinaryTF_Url_4,BinaryTF_Url_5,BinaryTF_Url_6,BinaryTF_Url_7,BinaryTF_Url_8,BinaryTF_Url_9,BinaryTF_Title_0,BinaryTF_Title_1,BinaryTF_Title_2,BinaryTF_Title_3,BinaryTF_Title_4,BinaryTF_Title_5,BinaryTF_Title_6,BinaryTF_Title_7,BinaryTF_Title_8,BinaryTF_Title_9,BinaryTF_Body_0,BinaryTF_Body_1,BinaryTF_Body_2,BinaryTF_Body_3,BinaryTF_Body_4,BinaryTF_Body_5,BinaryTF_Body_6,BinaryTF_Body_7,BinaryTF_Body_8,BinaryTF_Body_9,AnchorFlatStreamLength,AnchorTotalPhraseCount,BodyTermCount,DomainRank,StaticRank,StreamLength_Anchor,StreamLength_Body,StreamLength_Title,StreamLength_Url,TbDomainUsers,WordsInDomain,WordsInPath,WordsInTitle,LanguagePreference,LocationPreference,d,TermShardTotalQueryTermCoverage,TermShardTotalQueryTermCoverageWeighted,TermShardMatchedTermCountWeighted_AUTB,TermShardMatchedTermCountWeighted_Click'.split(',')
    mean, std = Test1(feature_list)
    Test2(feature_list, mean, std)

