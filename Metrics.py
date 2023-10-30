import pandas as pd
import numpy as np
import time

class Metrics(object):

    @staticmethod
    def calc_ndcg(idealset, recallset, topn):
        idealset = idealset[['QueryID', 'UrlHash16B', 'Rating']].drop_duplicates()
        recallset = recallset[recallset['IsMainPath']==1]
        recallset = recallset[recallset['RankPosition'] <= topn]
        recallset = recallset[['QueryID', 'UrlHash16B', 'RankPosition']].drop_duplicates()

        recallset = pd.merge(idealset, recallset, on=['QueryID', 'UrlHash16B'])


        # idealset = idealset.groupby(['QueryID'], as_index=False)['Rating'].sum()
        # recallset = recallset.groupby(['QueryID'], as_index=False)['Rating'].sum()

        #---
        idealset['RankPosition']  = idealset.groupby(['QueryID'])['Rating'].rank(ascending=0,method='first')


        idealset['Discount'] = 1.0 / np.log(idealset['RankPosition'] + 1)
        idealset['DCG'] = idealset['Rating'] * idealset['Discount']
        idealset = idealset.groupby(['QueryID'], as_index=False)['DCG'].sum()

        recallset['Discount'] = 1.0 / np.log(recallset['RankPosition'] + 1)
        recallset['DCG'] = recallset['Rating'] * recallset['Discount']
        recallset = recallset.groupby(['QueryID'], as_index=False)['DCG'].sum()

        join = pd.merge(idealset, recallset, on='QueryID', how='left')
        join['NDCG'] = join['DCG_y'] / join['DCG_x'] * 100

        ndcg = join['NDCG'].sum() / join['QueryID'].count()
        print(ndcg)

    @staticmethod
    def calc_fidelity(idealset, recallset, main_path_quota=56, sub_path_quota=10):
        idealset = idealset[['QueryID', 'UrlHash16B', 'Rating']].drop_duplicates()

        # recallset['L1Rank'] = recallset['QDVectorSimilarity']
        recallset['RankPosition']  = recallset.groupby(['QueryID', 'SubID'])['L1Rank'].rank(ascending=0,method='first')
        main_path = recallset.loc[recallset['IsMainPath']==1].loc[recallset['RankPosition'] <= main_path_quota]
        sub_path = recallset.loc[recallset['IsMainPath']==0].loc[recallset['RankPosition'] <= sub_path_quota]
        recallset = pd.concat([main_path, sub_path], axis=0)
        # recallset = recallset[recallset['QueryID']==1000879061]
        # print(recallset[['QueryID', 'UrlHash16B', 'SubID', 'QDVectorSimilarity', 'RankPosition']])


        recallset = recallset[['QueryID', 'UrlHash16B']].drop_duplicates()

        recallset = pd.merge(idealset, recallset, on=['QueryID', 'UrlHash16B'])


        idealset = idealset.groupby(['QueryID'], as_index=False)['Rating'].sum()
        recallset = recallset.groupby(['QueryID'], as_index=False)['Rating'].sum()


        #---
        join = pd.merge(idealset, recallset, on='QueryID', how='left')
        join['Fidelity'] = join['Rating_y'] / join['Rating_x'] * 100

        fidelity = join['Fidelity'].sum() / join['QueryID'].count()
        return fidelity

    @staticmethod
    def calc_fidelity_segment(idealset, recallset, segment, main_path_quota=56, sub_path_quota=10):

        idealset = idealset[['QueryID', 'UrlHash16B', segment, 'Rating']].drop_duplicates()

        recallset['RankPosition']  = recallset.groupby(['QueryID', 'SubID'])['L1Rank'].rank(ascending=0,method='first')
        main_path = recallset.loc[recallset['IsMainPath']==1].loc[recallset['RankPosition'] <= main_path_quota]
        sub_path = recallset.loc[recallset['IsMainPath']==0].loc[recallset['RankPosition'] <= sub_path_quota]
        recallset = pd.concat([main_path, sub_path], axis=0)
        recallset = recallset[['QueryID', 'UrlHash16B']].drop_duplicates()

        recallset = pd.merge(idealset, recallset, on=['QueryID', 'UrlHash16B'])


        idealset = idealset.groupby(['QueryID', segment], as_index=False)['Rating'].sum()
        recallset = recallset.groupby(['QueryID', segment], as_index=False)['Rating'].sum()


        #---
        join = pd.merge(idealset, recallset, on=['QueryID', segment], how='left')
        join['Fidelity'] = join['Rating_y'] / join['Rating_x'] * 100

        fidelity_by_segment = join.groupby([segment])['Fidelity'].sum() / join.groupby([segment])['QueryID'].count()



if __name__ == '__main__':
    idealset = pd.read_csv('./Data/EvalIdealset.tsv', sep='\t')
    # recallset = pd.read_csv('./Data/recallset.tsv', sep='\t', names=['QueryID','SubID','IsMainPath','UrlHash16B','L1Rank','RankPosition'])
    # recallset = pd.read_csv('./Data/recallset2.tsv', sep='\t', names=['QueryID','SubID','IsMainPath','UrlHash16B','L1Rank','RankPosition'])
    # recallset = pd.read_csv('./Data/EvalData.tsv', sep='\t')
    recallset = pd.read_csv('./Data/EvalDataWithRating.tsv', sep='\t')




    fidelity = Metrics.calc_fidelity(idealset, recallset)
    fidelity_by_segment = Metrics.calc_fidelity_segment(idealset, recallset, 'Tier')
    print(fidelity)
    print(fidelity_by_segment)
    # Metrics.calc_ndcg(idealset, recallset, 10)
    # Metrics.calc_ndcg(idealset, recallset, 5)
