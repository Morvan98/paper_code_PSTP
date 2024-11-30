import pandas as pd 
import numpy as np
from tools import *
import random
from collections import Counter
from tqdm import tqdm
import joblib
random.seed(10160)
np.random.seed(7110)
####### precompute feature matrix
# ##########################################
import embedding_metrics.ALBATROSS_idr as albatross
import embedding_metrics.esm2_8m_embedding as esm2_8m
from model.ann import *

independent_validation_data = pd.read_csv(
    'data_processed/ps_propensity_prediction_evaluation/independent_test/independent_train_vali_seqs_all_dropdup.csv',sep=',')

print(Counter(independent_validation_data['category'].values))
#############precompting matrix for train and cross validation data
sequences = [
    check_seq_tool(seq) for seq in independent_validation_data['seq'].values]
len_seqs = [len(s) for s in sequences]
data_matrix_pth = 'data_processed/ps_propensity_prediction_evaluation/independent_test/data_matrix/'

alba_matrix_lst = []
for alba_modelname in ['asphericity','scaled_rg','scaled_re']:
    emb_lst = albatross._get_pos_wide_embedding_by_batch(
        sequences,alba_modelname)
    for e in emb_lst:
        alba_matrix_lst.append(e)
data_size = len(sequences)
alba_mergedmatrix_lst = []
for matrix1,matrix2,matrix3 in zip(
    alba_matrix_lst[:data_size],
    alba_matrix_lst[data_size:2*data_size],
    alba_matrix_lst[2*data_size:3*data_size]):
    merged_matrix = np.concatenate((matrix1,matrix2,matrix3),axis=1)
    alba_mergedmatrix_lst.append(merged_matrix)
joblib.dump(alba_mergedmatrix_lst,f'{data_matrix_pth}/seqs_alba_poswide_matrix_lst.joblib')

esm_pos_wide_matrix = esm2_8m.get_seq_poswide_embedding_matrix(sequences)
print(esm_pos_wide_matrix)
joblib.dump(esm_pos_wide_matrix,f'{data_matrix_pth}/seqs_esm8m_poswide_matrix_lst.joblib')
# ######################################################################
######################################################################
############## training on non-human test on human
print('loading full dataset')
esm_pos_wide_matrix_lst = joblib.load(f'{data_matrix_pth}/seqs_esm8m_poswide_matrix_lst.joblib',)
alba_pos_wide_matrix_lst = joblib.load(f'{data_matrix_pth}/seqs_alba_poswide_matrix_lst.joblib',)
print('merging dataset')
full_matrixs_lst = []
for esm_matrix,alba_matrix in tqdm(zip(list(esm_pos_wide_matrix_lst),
                                  list(alba_pos_wide_matrix_lst))):
    merged_esm_alba = np.concatenate((esm_matrix,alba_matrix),axis=1)
    print(merged_esm_alba.shape)
    full_matrixs_lst.append(merged_esm_alba)
    
print(len(full_matrixs_lst))
###########################################
###########################################
###########################################
print(independent_validation_data.head(10))
categories = ['hps167','mixps237','mixps488','nonhps5754','nops']
def cate2label(category):
    if category in ['hps167','mixps237','mixps488',]:
        return 1
    else:
        return 0
independent_validation_data['label'] = independent_validation_data['category'].apply(lambda x:cate2label(x))

sample_total_idx = np.array([idx for idx in range(independent_validation_data.shape[0])])
independent_validation_data['sample_original_idx'] = sample_total_idx
category = independent_validation_data['category'].values
labels = independent_validation_data['label'].values
print(len(labels))
##### #### KEEP traing set non-human
nonhps_sample_idx = independent_validation_data[(independent_validation_data['is_human']==0)
                                   &(independent_validation_data['label']==1)]['sample_original_idx'].values
hps167_samples_idx = sample_total_idx[np.where(category=='hps167')]
mixps237_samples_idx = sample_total_idx[np.where(category=='mixps237')]
mixps488_samples_idx = sample_total_idx[np.where(category=='mixps488')]
nonhps5754_samples_idx = sample_total_idx[np.where(category=='nonhps5754')]
nops_samples_idx = sample_total_idx[np.where(category=='nops')]
train_cv_data_matrix = np.array(full_matrixs_lst,dtype=object)
############# validation on human ps proteins
train_positive_dataset_idx = nonhps_sample_idx #### KEEP traing set non-human
test_positive_dataset_idx = hps167_samples_idx
print('training/test positive datasize')
print(len(train_positive_dataset_idx),len(test_positive_dataset_idx))

val_auc_lst,val_aupr_lst = [],[]
val_py_lst_lst,val_ty_lst_lst,val_idx_lst_lst = [],[],[]
iter = 20

for i__ in tqdm(range(iter)): 
    train_nops_nonhuman_sample_subset_idx = nops_samples_idx.copy()
    train_merge_pos_neg_sample_idx_list = np.array(
        list(train_positive_dataset_idx)+list(train_nops_nonhuman_sample_subset_idx)\
            )
    train_idx_lst = train_merge_pos_neg_sample_idx_list.copy()
    np.random.shuffle(train_idx_lst)
    train_x = train_cv_data_matrix[train_idx_lst]
    train_y = labels[train_idx_lst]
    m = sliding_nn(650)
    train_slidenn_model(m,train_x,train_y,iter=50,l1=0.0001,sample_neg=True,gfs=True)
    
    test_nops_human_sample_subset_idx = nonhps5754_samples_idx
    test_merge_pos_neg_sample_idx_list = np.array(
        list(test_positive_dataset_idx)+\
            list(test_nops_human_sample_subset_idx))
    test_idx_lst = test_merge_pos_neg_sample_idx_list.copy()
    # np.random.shuffle(test_idx_lst)
    test_x = train_cv_data_matrix[test_idx_lst]
    test_y = labels[test_idx_lst]
    py = slidenn_prediction(m,test_x)
    bauc,baupr = balanced_auc_aupr(test_y,py)
    auc_ = roc_auc_score(test_y,py)
    aupr_ = compute_aupr(test_y,py)
    print('training/testing class weights:',Counter(train_y),Counter(test_y))
    print('balanced auc/aupr',
        bauc,baupr)
    
    val_auc_lst.append(auc_)
    val_aupr_lst.append(baupr)
    val_ty_lst_lst.append(test_y)
    val_py_lst_lst.append(py)
    val_idx_lst_lst.append(test_idx_lst)
    print(Counter(test_y),len(test_y),len(py),len(test_idx_lst))
    print('---- current overall performances ----')
    print('validation auc median/average/max/min',np.median(val_auc_lst),np.average(val_auc_lst),
    np.max(val_auc_lst),np.min(val_auc_lst))
    print('validation aupr median/average/median/max/min',np.average(val_aupr_lst),
        np.median(val_aupr_lst),np.max(val_aupr_lst),np.min(val_aupr_lst))

    del m,bauc,baupr
    

