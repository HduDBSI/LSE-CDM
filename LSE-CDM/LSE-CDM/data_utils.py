import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path, seq_length):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        uid_max = max(uid_max, uid)
        iid_max = max(iid_max, iid)

    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    # 构建交互稀疏矩阵
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
        (train_list[:, 0], train_list[:, 1])), dtype='float64',
        shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
        (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
        shape=(n_user, n_item))

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
        (test_list[:, 0], test_list[:, 1])), dtype='float64',
        shape=(n_user, n_item))

    # 构造用户交互序列和长度（保留最后 seq_length 个）
    train_seqs, train_seq_lens, train_inter_counts = [], [], []
    for u in range(n_user):
        items = train_dict.get(u, [])
        real_count = len(items)  # 实际交互数
        train_inter_counts.append(real_count)
        if len(items) > seq_length:
            items = items[-seq_length:]  # 截断
        train_seqs.append(np.array(items, dtype=np.int32))
        train_seq_lens.append(len(items))  # 截断长度

    train_seq_lens = np.array(train_seq_lens, dtype=np.int32)
    train_inter_counts = np.array(train_inter_counts, dtype=np.int32)
    train_seqs_array = np.array(train_seqs, dtype=object)

    return train_data, valid_y_data, test_y_data, \
           train_seqs_array, train_seq_lens, train_inter_counts, n_user, n_item



def subdata_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0

    train_dict = {}
    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1

    valid_dict = {}
    for uid, iid in valid_list:
        if uid not in valid_dict:
            valid_dict[uid] = []
        valid_dict[uid].append(iid)
    
    test_dict = {}
    for uid, iid in test_list:
        if uid not in test_dict:
            test_dict[uid] = []
        test_dict[uid].append(iid)
    
    return train_dict, valid_dict, test_dict, n_user, n_item

class SubData(Dataset):
    def __init__(self, train_path, valid_path, test_path, num_sub=1000):
        super(SubData, self).__init__()
        self.train_dict, self.valid_dict, self.test_dict, \
            self.num_user, self.num_item = subdata_load(train_path, valid_path, test_path)
        self.num_sub = num_sub
        
        self.item_set = set(range(0, self.num_item))

        self.all_user = [i for i in range(self.num_user)]

        self.val_list, self.val_gt = self.get_val(self.valid_dict)
        self.test_list, self.test_gt = self.get_test(self.test_dict)

    def get_val(self, data):
        # data: ground truth
        val_list = [[] for _ in range(self.num_user)]
        gt_list = [[] for _ in range(self.num_user)]

        for uid in data:
            val_list[uid].extend(data[uid])
            gt_list[uid].extend([i for i in range(len(data[uid]))])
            try:
                a = self.item_set - set(self.train_dict[uid])  # mask train set
            except:
                if uid not in self.train_dict:
                    print("User not found.")
                print("Error!")
            m = np.random.choice(np.array(list(a)), self.num_sub-len(val_list[uid]), replace=False)
            val_list[uid].extend(m)
        
        for i in range(len(val_list)):
            if len(val_list[i]) == 0:
                val_list[i] = [0] * self.num_sub
        
        val_list = torch.LongTensor(val_list)
        return val_list, gt_list
    
    def get_test(self, data):
        test_list = [[] for _ in range(self.num_user)]
        gt_list = [[] for _ in range(self.num_user)]

        for uid in data:
            test_list[uid].extend(data[uid])
            gt_list[uid].extend([i for i in range(len(data[uid]))])
            try:
                m = np.random.choice(np.array(list(self.item_set-set(self.train_dict[uid])-set(self.valid_dict[uid]))), self.num_sub-len(test_list[uid]), replace=False)
            except:
                m = np.random.choice(np.array(list(self.item_set-set(self.train_dict[uid]))), self.num_sub-len(test_list[uid]), replace=False)
            test_list[uid].extend[m]

        for i in range(len(test_list)):
            if len(test_list[i]) == 0:
                test_list[i] = [0] * self.num_sub
        
        test_list = torch.LongTensor(test_list)
        return test_list, gt_list
            

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)

class FullTrainDataset(Dataset):
    def __init__(self, train_data, train_seqs, train_seq_lens, train_inter_counts):
        self.train_data = train_data
        self.train_seqs = train_seqs
        self.train_seq_lens = train_seq_lens
        self.train_inter_counts = train_inter_counts  # 新增

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return {
            'dense': self.train_data[idx],
            'seq': torch.LongTensor(self.train_seqs[idx]),
            'len': torch.tensor(self.train_seq_lens[idx], dtype=torch.long),
            'real_len': torch.tensor(self.train_inter_counts[idx], dtype=torch.long)  # 新增
        }


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    dense_batch = torch.stack([item['dense'] for item in batch], dim=0)
    seq_batch = [item['seq'] for item in batch]
    len_batch = torch.tensor([item['len'] for item in batch], dtype=torch.long)
    real_len_batch = torch.tensor([item['real_len'] for item in batch], dtype=torch.long)  # 新增

    seq_batch_padded = pad_sequence(seq_batch, batch_first=True, padding_value=0)

    return {
        'dense': dense_batch,
        'seq': seq_batch_padded,
        'len': len_batch,
        'real_len': real_len_batch
    }

