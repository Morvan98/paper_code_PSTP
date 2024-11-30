from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
import joblib 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import requests

'''
preprocessing
'''
RESIDUES_SET = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

def get_label_from_sig(sig):
    if sig in ['Pathogenic','Likely pathogenic']:
        return 1
    elif sig in ['Benign','Likely benign']:
        return 0
    else:
        print(sig)
        
def check_seq_tool(seq):
    seq = str(seq)
    if set(list(seq)) == RESIDUES_SET:
        return seq
    else:
        new_seq = ''.join([aa if aa in RESIDUES_SET else 'A' for aa in seq])
        return new_seq
    
'''
results analyze
'''
def reverse_index_map(before_lst,after_lst):
    '''
    before_lst: orginal index lst [1,2,3,4,5,6]
    after_lst : index lst shuffled [2,3,1,6,5,4]
    '''
    map_dict = {}
    for idx in range(len(after_lst)):
        map_dict[after_lst[idx]] = idx
    return_index = [map_dict[x] for x in before_lst]
    return return_index

# print(reverse_index_map([1,2,3,4,5,6],[2,3,1,6,5,4])) ## return [2, 0, 1, 5, 4, 3] 

def get_best_cutoff(ty,py):
    ### get best cutoff according to f1_score
    cutoff = [0.05*x for x in range(20)]
    f1s = []
    for cut in cutoff:
        py_binary = np.where(np.array(py)>cut,1,0)
        f1s.append(f1_score(ty,py_binary))
    return cutoff[np.argmax(f1s)]
def compute_aupr(y_true,y_predicted):
    precision, recall, _ = precision_recall_curve(y_true, y_predicted)
    return auc(recall,precision)


def balanced_auc_aupr(y_true,y_predicted):
    '''
    y_true,y_predicted as input
    '''
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    label_0_indices = np.where(y_true == 0)
    label_1_indices = np.where(y_true == 1)
    py_label0 = np.array(y_predicted)[label_0_indices]
    py_label1 = np.array(y_predicted)[label_1_indices]
    sample_size_ = int(min(len(py_label0),len(py_label1)) * 0.8)
    total_auc,total_aupr = 0,0
    for _ in range(100):
        py_label0_sampled = np.random.choice(py_label0, size=sample_size_, replace=False)
        py_label1_sampled = np.random.choice(py_label1, size=sample_size_, replace=False)
        py_sampled = list(py_label0_sampled)+list(py_label1_sampled)
        ty_sampled = [0]*len(py_label0_sampled)+[1]*len(py_label1_sampled)
        total_auc += roc_auc_score(ty_sampled,py_sampled) 
        total_aupr += compute_aupr(ty_sampled,py_sampled) 
    return total_auc/100,total_aupr/100


'''
encoding related
'''
RESIDUES_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_index = {aa: index for index, aa in enumerate(RESIDUES_list)}
def create_one_hot_for_row(row):
    '''
    def create_one_hot_for_row(row):
    one_hot_wt = np.zeros(len(RESIDUES_list))
    one_hot_mt = np.zeros(len(RESIDUES_list))
    one_hot_wt[aa_to_index[row['wt_aa']]] = 1
    one_hot_mt[aa_to_index[row['mt_aa']]] = 1
    '''
    one_hot_wt = np.zeros(len(RESIDUES_list))
    one_hot_mt = np.zeros(len(RESIDUES_list))
    one_hot_wt[aa_to_index[row['wt_aa']]] = 1
    one_hot_mt[aa_to_index[row['mt_aa']]] = 1
    one_hot = np.concatenate([one_hot_wt, one_hot_mt])
    
    return one_hot
def create_one_hot_vec(wt_aa,mt_aa):
    '''
    create_one_hot_vec(wt_aa,mt_aa)
    '''
    one_hot_wt = np.zeros(len(RESIDUES_list))
    one_hot_mt = np.zeros(len(RESIDUES_list))
    one_hot_wt[aa_to_index[wt_aa]] = 1
    one_hot_mt[aa_to_index[mt_aa]] = 1
    one_hot = np.concatenate([one_hot_wt, one_hot_mt])
    return one_hot

def create_one_hot_matrix(wt_aa_lst,mt_aa_lst):
    '''
    # create_one_hot_matrix(wt_aa_list,mt_aa_list)
    '''
    one_hot_matrix = np.zeros(len(wt_aa_lst),40)
    for i_,(wt_aa,mt_aa) in enumerate(zip(wt_aa_lst,mt_aa_lst)):
        one_hot_aa = create_one_hot_vec(wt_aa,mt_aa)
        one_hot_matrix[i_,:] = one_hot_aa
    return one_hot_matrix



prot_vec_dict = joblib.load('embedding_metrics/prot_vec_dict.pkl')

def prot_vec_map(mer3):
    try:
        return prot_vec_dict[mer3]
    except:
        return prot_vec_dict['AAA'] 

def protVecEncode_avg(seq):
    assert len(seq) >= 3
    if len(seq) == 3:
        return np.array(prot_vec_map(seq))
    return np.sum([prot_vec_map(seq[idx:idx+3]) for idx in range(0,len(seq)-3)],axis=0)/(len(seq)-3)

def protVecEncode_sum(seq):
    assert len(seq) >= 3
    if len(seq) == 3:
        return np.array(prot_vec_map(seq))
    return np.sum([prot_vec_map(seq[idx:idx+3]) for idx in range(0,len(seq)-3)],axis=0)


def protVecEncode_seqlist_input(seq_lst):
    output = []
    for s in seq_lst:
        output.append(protVecEncode_avg(s))
    return np.array(output)


'''
compute windows
'''
def _find_continuous_segments(lst, value_=1):
    segments = []
    start = None
    for i, v in enumerate(lst):
        if v == value_:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append([start, i-1])
                start = None
    if start is not None:
        segments.append([start, len(lst)-1])
    return segments
def compute_window(score_list,threshold,min_win_len=10,adjust_idx=1):
    '''
    example:
    # test_score_lst = [50,50,50,65,65,68,89,80,80,80,50,50,80,40,40,40,40,60]
    # print(compute_window(test_score_lst,70,5))
    # return [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    # and a list like [[...],[...]]
    # print(compute_window([0.3,0.2,0.4,0.1,0.2,0.6,0.6,0.6,0.7,0.8,0.4,0.3],0.5,5,-1))
    # ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [[5, 11]])
    # 
    '''
    assert adjust_idx in [1,-1] # this index is to adjust the direction of comparison
    n = len(score_list)
    win_marks = [0] * n
    # Step 1: Mark initial regions
    for i in range(n):
        if (score_list[i] - threshold) * adjust_idx < 0:
            count = 1
            while i + count < n and (score_list[i + count] - threshold) * adjust_idx < 0:
                count += 1
            if count >= min_win_len:
                for j in range(i, i + count + 1):
                    j = min(j,n-1)
                    win_marks[j] = 1     
    # Step 2: Reclassify short regions between wins as wins
    i = 0
    while i < n:
        if win_marks[i] == 0:
            start = i
            while i < n and win_marks[i] == 0:
                i += 1
            end = i
            if end - start < min_win_len:
                if (start == 0 or win_marks[start - 1] == 1) and (end == n or win_marks[end] == 1):
                    for j in range(start, end):
                        win_marks[j] = 1
        i += 1   
    win_regions = _find_continuous_segments(win_marks)  
    return win_marks,win_regions

'''
get local flucturation
'''
def compute_local_flucturation(score_lst,win_size=11):
    dsize = len(score_lst)
    assert dsize > win_size
    std_lst = []
    for idx_ in range(dsize):
        win_start = min(dsize-win_size+1,idx_-win_size//2)
        win_start = max(win_start,0)
        win_end = max(win_size-1,idx_+win_size//2)
        # print(win_start,win_end)
        window = score_lst[win_start:win_end]
        # print(window)
        std_lst.append(np.array(window).std())
        # std_lst.append(abs(max(window)-min(window)))
    return std_lst
# print(compute_local_flucturation(
#     [1,1,1,1,1,2,2,2,3,42,4,2,4,2,5,6,1,1,1,1,1,2,2,2,]
# ))

'''
write fasta file
'''
def write_seq_to_fasta(seq_name_lst,seq_lst,save_pth):
    seq_records = []
    for record_name,seq in zip(seq_name_lst,seq_lst):
        if seq == 'not matched':
            continue
        # print(record_name)
        seq_records.append(SeqRecord(
        Seq(str(seq)),
        id=record_name,
        description=''
    ) )
    # print(f'writing {len(seq_records)} seqs')
    with open(save_pth, 'w') as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")
''' 
read fasta
'''
def get_seqid_seq_from_fasta(fa_pth):
    seqid_lst,seq_lst = [],[]
    for record in SeqIO.parse(fa_pth,'fasta'):
        try:
            seqid_ = record.description
            seqid_lst.append(seqid_)
            seq_lst.append(str(record.seq))
        except:
            pass
    return seqid_lst,seq_lst
'''
get seq by url
'''
def fetch_uniprot_sequence_from_url(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    # "A0A0C9PCY9"
    try:
        response = requests.get(url,verify=False)
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])
        return sequence
    except:
        print(f"Failed to fetch sequence for UniProt ID: {uniprot_id}")
        return 'not matched'
