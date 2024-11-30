import embedding_metrics.ALBATROSS_idr as albatross
import embedding_metrics.esm2_8m_embedding as esm2_8m
from tools import *
from model.ann import *
import os
def seq2matrix_lst(full_sequences):
    sequences = [
        check_seq_tool(seq) for seq in full_sequences]
    esm_pos_wide_matrix_lst = esm2_8m.get_seq_poswide_embedding_matrix(sequences)
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
    print('merging dataset')
    full_matrixs_lst = []
    for esm_matrix,alba_matrix in tqdm(zip(list(esm_pos_wide_matrix_lst),
                                    list(alba_mergedmatrix_lst))):
        merged_esm_alba = np.concatenate((esm_matrix,alba_matrix),axis=1)
        # print(merged_esm_alba.shape)
        full_matrixs_lst.append(merged_esm_alba)
    return np.array(full_matrixs_lst)  

def matrixlst_to_matrix(matrix_lst):
    '''
    [matrix1,matrix2,..] -> [vec,vec,vec,] vec is size 650
    each matrix size-> [seq_len of n,650]
    to size 650 vector
    '''
    array_lst = []
    for matrix in matrix_lst:
        avg_vec = np.average(matrix,axis=0)
        array_lst.append(avg_vec)
    return np.array(array_lst)
    
'''
load saps/pdps/mixdata trained models 
'''
saps_models_pth ='model/slide_nn_model_weights/trained_on_phasepred_saps'
saps_models_pth_nophasepro ='model/slide_nn_model_weights/trained_on_phasepred_saps_nophasepro'
saps_models_pth_notrunc = 'model/slide_nn_model_weights/trained_on_phasepred_saps_notrunc'


pdps_models_pth = 'model/slide_nn_model_weights/trained_on_phasepred_pdps'
pdps_models_pth_nophasepro = 'model/slide_nn_model_weights/trained_on_phasepred_pdps_nophasepro'
pdps_models_pth_notrunc = 'model/slide_nn_model_weights/trained_on_phasepred_pdps_notrunc'

mix_models_pth = 'model/slide_nn_model_weights/trained_on_psphunter'
mix_models_pth_nophasepro = 'model/slide_nn_model_weights/trained_on_psphunter_nophasepro'
mix_models_pth_notrunc = 'model/slide_nn_model_weights/trained_on_psp_hunter_notruncated'

def _get_model_from_pth(weight_pth):
    m_ = sliding_nn(650)
    state_dict = torch.load(weight_pth)
    m_.load_state_dict(state_dict)
    m_.eval()
    return m_

def _get_fixed_model_from_pth(weight_pth):
    m_ = sliding_nn_fixed(650)
    state_dict = torch.load(weight_pth)
    m_.load_state_dict(state_dict)
    m_.eval()
    return m_

def _get_kernel_from_pth(weight_pth):
    m_ = kernel_only(650)
    state_dict = torch.load(weight_pth)
    m_.load_state_dict(state_dict)
    m_.eval()
    return m_

saps_models = [_get_model_from_pth(f'{saps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
saps_fixed_models = [_get_fixed_model_from_pth(f'{saps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
saps_models_nophasepro = [_get_model_from_pth(f'{saps_models_pth_nophasepro}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,5)]
saps_kernels = [_get_kernel_from_pth(f'{saps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
saps_kernels_notrunc = [_get_kernel_from_pth(f'{saps_models_pth_notrunc}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]

pdps_models = [_get_model_from_pth(f'{pdps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
pdps_fixed_models = [_get_fixed_model_from_pth(f'{pdps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
pdps_models_nophasepro = [_get_model_from_pth(f'{pdps_models_pth_nophasepro}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,5)]
pdps_kernels = [_get_kernel_from_pth(f'{pdps_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
pdps_kernels_notrunc = [_get_kernel_from_pth(f'{pdps_models_pth_notrunc}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]

mix_models = [_get_model_from_pth(f'{mix_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
mix_fixed_models = [_get_fixed_model_from_pth(f'{mix_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
mix_models_nophasepro = [_get_model_from_pth(f'{mix_models_pth_nophasepro}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,5)]
mix_kernels = [_get_kernel_from_pth(f'{mix_models_pth}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]
mix_kernels_notrunc = [_get_kernel_from_pth(f'{mix_models_pth_notrunc}/dense_650_20_5_1_weights_{idx}.pth') for idx in range(0,10)]



def predict_by_saps_models(seq_embedding_matrix,predict_phasepro=False,
                           winsize=33):
    if predict_phasepro:
        _saps_models = saps_models
    else:
        _saps_models = saps_models_nophasepro
    x = torch.tensor(seq_embedding_matrix
                     ,dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    win_scores = [m_.get_window_score(x,winsize) for m_ in _saps_models]
    win_score = np.average(win_scores,axis=0)
    assert len(win_score) == x.shape[1]
    predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in _saps_models]
    predicted_score = np.average(predicted_scores)
    return win_score,predicted_score

def predict_by_pdps_models(seq_embedding_matrix,predict_phasepro=False,
                           winsize=33):
    if predict_phasepro:
        _pdps_models = pdps_models
    else:
        _pdps_models = pdps_models_nophasepro
    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    win_scores = [m_.get_window_score(x,winsize) for m_ in _pdps_models]
    win_score = np.average(win_scores,axis=0)
    assert len(win_score) == x.shape[1]
    predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in _pdps_models]
    predicted_score = np.average(predicted_scores)
    return win_score,predicted_score

def predict_by_mix_models(seq_embedding_matrix,predict_phasepro=False,
                          winsize=33):
    if predict_phasepro:
        _mix_models = mix_models
    else:
        _mix_models = mix_models_nophasepro

    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    win_scores = [m_.get_window_score(x,winsize) for m_ in _mix_models]
    win_score = np.average(win_scores,axis=0)
    assert len(win_score) == x.shape[1]
    predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in _mix_models]
    predicted_score = np.average(predicted_scores)
    return win_score,predicted_score

def saps_max(seq_embedding_matrix,winsize=33):
    x = torch.tensor(seq_embedding_matrix
                     ,dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    predicted_scores = [m_.forward(x,winsize,scale=False).detach().cpu().numpy() for m_ in saps_fixed_models]
    predicted_score = np.average(predicted_scores)
    return predicted_score

def pdps_max(seq_embedding_matrix,winsize=33):
    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    predicted_scores = [m_.forward(x,winsize,scale=False).detach().cpu().numpy() for m_ in pdps_fixed_models]
    predicted_score = np.average(predicted_scores)
    return predicted_score

def mix_max(seq_embedding_matrix,winsize=33):
    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    predicted_scores = [m_.forward(x,winsize,scale=False).detach().cpu().numpy() for m_ in mix_fixed_models]
    predicted_score = np.average(predicted_scores)
    return predicted_score

def saps_kernel(seq_embedding_matrix,trunc_data=False):
    x = torch.tensor(seq_embedding_matrix
                     ,dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    if trunc_data:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in saps_kernels_notrunc]
    else:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in saps_kernels]
    
    predicted_score = np.average(predicted_scores)
    return predicted_score

def pdps_kernel(seq_embedding_matrix,trunc_data=False):
    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    if trunc_data:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in pdps_kernels_notrunc]
    else:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in pdps_kernels]
    predicted_score = np.average(predicted_scores)
    return predicted_score

def mix_kernel(seq_embedding_matrix,trunc_data=False):
    x = torch.tensor(seq_embedding_matrix,
                     dtype=torch.float).unsqueeze(0)
    assert x.shape[0] == 1
    if trunc_data:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in mix_kernels_notrunc]
    else:
        predicted_scores = [m_.forward(x).detach().cpu().numpy() for m_ in mix_kernels]
    predicted_score = np.average(predicted_scores)
    return predicted_score
