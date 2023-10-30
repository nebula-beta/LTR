import os
from typing import Tuple, Optional
import math
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Utils import init_logger, get_logger, rating_to_label_gains


class FeatureInfo(object):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.linear_min = None
        self.linear_max = None
        self.linear_bucket_size = None
        self.linear_mean = None
        self.linear_std = None
        #self.linear_var = None
        self.linear_entropy = None
        self.linear_slope = None
        self.linear_bias = None

        self.log_min = None
        self.log_max = None
        self.log_bucket_size = None
        self.log_mean = None
        self.log_std = None
        #self.log_var = None
        self.log_entropy = None
        self.log_slope = None
        self.log_bias = None

        self.total_values = None
        self.zero_values = None
        self.unique_count = 0
        pass

    def __repr__(self):
        output = "FeatureInfo(feature_name={}, linear_min={:.5f}, linear_max=:{:.5f}, linear_bucket_size={:.5f}, linear_mean={:.5f}, linear_std={:.5f}, linear_entropy={:.5f}, linear_slope={:.5f}, linear_bias={:.5f}, log_min={:.5f}, log_max=:{:.5f}, log_bucket_size={:.5f}, log_mean={:.5f}, log_std={:.5f}, log_entropy={:.5f}, log_slope={:.5f}, log_bias={:.5f}, total_values={:.5f}, zero_values={:.5f}, unique_count={:.5f})".format(self.feature_name, self.linear_min, self.linear_max, self.linear_bucket_size, self.linear_mean, self.linear_std, self.linear_entropy, self.linear_slope, self.linear_bias, self.log_min, self.log_max, self.log_bucket_size, self.log_mean, self.log_std, self.log_entropy, self.log_slope, self.log_bias, self.total_values, self.zero_values, self.unique_count)
        return output




def FeatureInfo_repr(dumper, data):
    return dumper.represent_mapping(u"!FeatureInfo", {"feature_name" : data.feature_name, "linear_min" : data.linear_min, "linear_max" : data.linear_max, "linear_bucket_size" : data.linear_bucket_size,  "linear_mean" : data.linear_mean, "linear_std" : data.linear_std, "linear_entropy" : data.linear_entropy, "linear_slope" : data.linear_slope, "linear_bias" : data.linear_bias, "log_min" : data.log_min, "log_max" : data.log_max, "log_bucket_size" : data.log_bucket_size,  "log_mean" : data.log_mean, "log_std" : data.log_std, "log_entropy" : data.log_entropy, "log_slope" : data.log_slope, "log_bias" : data.log_bias, "total_values" : data.total_values, "zero_values" : data.zero_values, "unique_count" : data.unique_count})

def FeatureInfo_cons(loader, node):
    value = loader.construct_mapping(node)
    feature_name = value["feature_name"]

    linear_min = value["linear_min"]
    linear_max = value["linear_max"]
    linear_bucket_size = value["linear_bucket_size"]
    linear_mean = value["linear_mean"]
    linear_std = value["linear_std"]
    linear_entropy = value["linear_entropy"]
    linear_slope = value["linear_slope"]
    linear_bias = value["linear_bias"]


    log_min = value["log_min"]
    log_max = value["log_max"]
    log_bucket_size = value["log_bucket_size"]
    log_mean = value["log_mean"]
    log_std = value["log_std"]
    log_entropy = value["log_entropy"]
    log_slope = value["log_slope"]
    log_bias = value["log_bias"]

    total_values = value["total_values"]
    zero_values = value["zero_values"]
    unique_count = value["unique_count"]


    feature_info = FeatureInfo(feature_name)

    feature_info.linear_min = linear_min
    feature_info.linear_max = linear_max
    feature_info.linear_bucket_size = linear_bucket_size
    feature_info.linear_mean = linear_mean
    feature_info.linear_std = linear_std
    feature_info.linear_entropy = linear_entropy
    feature_info.linear_slope = linear_slope
    feature_info.linear_bias = linear_bias

    feature_info.log_min = log_min
    feature_info.log_max = log_max
    feature_info.log_bucket_size = log_bucket_size
    feature_info.log_mean = log_mean
    feature_info.log_std = log_std
    feature_info.log_entropy = log_entropy
    feature_info.log_slope = log_slope
    feature_info.log_bias = log_bias

    feature_info.total_values = total_values
    feature_info.zero_values = zero_values
    feature_info.unique_count = unique_count

    return feature_info

def get_entropy(data_df,columns = None):
    if (columns is None) and (data_df.shape[1] > 1) :
        raise "the dim of data_df more than 1, the columns must be not empty!"    
    # entropy
    pe_value_array = data_df[columns].unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(data_df[data_df[columns] == x_value].shape[0]) / data_df.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    
    return ent
    
num_bucket = 100
def analysis(tsv_file_path, feature_list, output_file_path):
    """
    :param tsv_file_path: dataset file path
    :param query_id_name: header name of query_id
    :param rating_name: header name of rating
    :param feature_list: list contain feature list
    :param filter_queries: whether to filter queries that have no relevant documents associated with them.
    :return: LibSVMDataset instantiated from given file
    """

    sep='\t'
    df = pd.read_csv(tsv_file_path, sep=sep)
    X = df[feature_list]

    feature_info_dict = dict()
    for feature in feature_list:
        #print(feature)
        feature_info = FeatureInfo(feature)

        linear_feature = X[feature]
        log_feature = X[feature].apply(lambda x : math.log(x + 1.0))


        feature_info.linear_min = linear_feature.min().item()
        feature_info.linear_max = linear_feature.max().item()
        feature_info.linear_mean = linear_feature.mean().item()
        feature_info.linear_std = linear_feature.std().item()

        feature_info.log_min = log_feature.min().item()
        feature_info.log_max = log_feature.max().item()
        feature_info.log_mean = log_feature.mean().item()
        feature_info.log_std = log_feature.std().item()

        if feature_info.linear_min == feature_info.linear_max:
            #print(feature)
            continue

        feature_info.linear_bucket_size = (feature_info.linear_max - feature_info.linear_min) / num_bucket
        if feature_info.linear_bucket_size < 1.0:
            feature_info.linear_bucket_size = 1.0
        feature_info.log_bucket_size = (feature_info.log_max - feature_info.log_min) / num_bucket

        linear_bucket_feature = linear_feature.apply(lambda x : int((x - feature_info.linear_min) / feature_info.linear_bucket_size))
        feature_info.linear_entropy = get_entropy(pd.DataFrame(linear_bucket_feature, columns=[feature]), feature).item()

        log_bucket_feature = log_feature.apply(lambda x : int((x - feature_info.log_min) / feature_info.log_bucket_size))
        feature_info.log_entropy = get_entropy(pd.DataFrame(log_bucket_feature, columns=[feature]), feature).item()


        feature_info.linear_slope = 1.0 / feature_info.linear_std
        feature_info.linear_bias = -feature_info.linear_mean / feature_info.linear_std

        feature_info.log_slope = 1.0 / feature_info.log_std
        feature_info.log_bias = -feature_info.log_mean / feature_info.log_std

        feature_info.total_values = len(linear_feature)
        feature_info.zero_values = (linear_feature == 0).sum().item()
        feature_info.unique_count = len(linear_feature.unique())

        feature_info_dict[feature] = feature_info


    yaml.add_representer(FeatureInfo, FeatureInfo_repr)
    yaml.add_constructor(u'!FeatureInfo', FeatureInfo_cons)

    with open(output_file_path, "w") as fo:
        fo.write(yaml.dump(feature_info_dict, allow_unicode=True))


    #with open("feature_info.yaml", "r") as fi:
        #xx = yaml.load(fi.read(), Loader=yaml.FullLoader)





if __name__ == '__main__':
    all_feature_list = 'QDVectorSimilarity,TermShardTotalQueryLength,TermShardMatchedTermCount_AUTB,TermShardMatchedTermCount_Click,TermShardMatchedPostingCount_AUTB,TermShardMatchedPostingCount_Click,TermShardMaxTermCountOfPosting_AUTB,TermShardMaxTermCountOfPosting_Click,TermShardQueryTermCoverage_AUTB,TermShardQueryTermCoverage_Click,TermShardMaxIDFOfPosting_AUTB,TermShardMaxIDFOfPosting_Click,TermShardAvgIDFOfPosting_AUTB,TermShardAvgIDFOfPosting_Click,TermShardMinPostingRatio_AUTB,TermShardMinPostingRatio_Click,BinaryTF_Url_0,BinaryTF_Url_1,BinaryTF_Url_2,BinaryTF_Url_3,BinaryTF_Url_4,BinaryTF_Url_5,BinaryTF_Url_6,BinaryTF_Url_7,BinaryTF_Url_8,BinaryTF_Url_9,BinaryTF_Title_0,BinaryTF_Title_1,BinaryTF_Title_2,BinaryTF_Title_3,BinaryTF_Title_4,BinaryTF_Title_5,BinaryTF_Title_6,BinaryTF_Title_7,BinaryTF_Title_8,BinaryTF_Title_9,BinaryTF_Body_0,BinaryTF_Body_1,BinaryTF_Body_2,BinaryTF_Body_3,BinaryTF_Body_4,BinaryTF_Body_5,BinaryTF_Body_6,BinaryTF_Body_7,BinaryTF_Body_8,BinaryTF_Body_9,AnchorFlatStreamLength,AnchorTotalPhraseCount,BodyTermCount,DomainRank,StaticRank,StreamLength_Anchor,StreamLength_Body,StreamLength_Title,StreamLength_Url,TbDomainUsers,WordsInDomain,WordsInPath,WordsInTitle,LanguagePreference,LocationPreference,d,TermShardTotalQueryTermCoverage,TermShardTotalQueryTermCoverageWeighted,TermShardMatchedTermCountWeighted_AUTB,TermShardMatchedTermCountWeighted_Click'.split(',')

    print(len(all_feature_list))

    analysis('./Data/TrainSmall.tsv', all_feature_list, './Data/TrainSmall_FeatureInfo.NumBucket_{}.yaml'.format(num_bucket))
