import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATA_DIR = 'data/internal'
TEST_DATA_DIR = 'benchmark/data'
CKPT_PATH = 'models/embedding-based/best.pt'


class Config:
    device = 'cpu'
    epochs = 20
    seed = 0
    batch_size = 128
    embedding_dim = 32
    hidden_size = 16
    dropout_rate = 0.5
    lr = 1e-3


def set_seed(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class MovieDataset(Dataset):
    def __init__(self, ratings, users, items):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, ids):
        ratings = self.ratings.iloc[ids]

        user_ids = ratings.user_id.astype('int')
        item_ids = ratings.item_id.astype('int')

        users = self.users.iloc[user_ids]
        items = self.items.iloc[item_ids]

        return {
            "ratings": torch.tensor(ratings.rating, dtype=torch.long),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "item_ids": torch.tensor(item_ids, dtype=torch.long),
            "users_info": torch.tensor(users.to_numpy(), dtype=torch.float),
            "items_info": torch.tensor(items.to_numpy(), dtype=torch.float),
        }


class RecSysModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, hidden_size, dropout_rate, n_item_features, n_user_features):
        super().__init__()

        self.user_embed = nn.Embedding(n_users, embedding_dim=embedding_dim)
        self.item_embed = nn.Embedding(n_items, embedding_dim=embedding_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim * 2 +
                             n_item_features + n_user_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 2, 1)

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, user_ids, item_ids, users_info, items_info):
        user_embeds = self.user_embed(user_ids)
        item_embeds = self.item_embed(item_ids)

        x = torch.cat([user_embeds, item_embeds,
                      users_info, items_info], dim=1)

        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x

class UserInfo:
    def __init__(self, user_id, age, gender, occupation, zip_code):
        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.occupation = occupation
        self.zip_code = zip_code
    def __repr__(self) -> str:
        return f'User (id {self.user_id}):\tage: {self.age},\tgender: {self.gender},\toccupation: {self.occupation}'


def dup_rows(a, indx, num_dups=1):
    return np.insert(a, [indx+1]*num_dups, a[indx], axis=0)

def get_user_watched_movies(user_id, test_data):
    watched_movies = test_data[test_data.user_id == user_id].item_id
    watched_movie_ids = watched_movies.to_list()

    watched_movies_with_rating = []

    for i, movie_id in enumerate(watched_movie_ids):
        rating = test_data[(test_data.item_id == movie_id) & (test_data.user_id == user_id)].rating.values[0]
        watched_movies_with_rating.append((movie_id, rating))

    item_ratings = sorted(watched_movies_with_rating,key=lambda x: x[1])
    
    return item_ratings
    
def get_watched_movies_recommendation(model, user_id, test_data, item_data, user_data):
    model.eval()

    watched_movies = test_data[test_data.user_id == user_id].item_id
    watched_movie_ids = watched_movies.to_list()

    users_info_one_row = user_data.iloc[user_id]

    users_info = dup_rows(users_info_one_row.to_numpy()[np.newaxis, ...],
                          0, len(watched_movie_ids)-1)

    user_ids = [user_id] * len(watched_movie_ids)

    items_info = np.zeros((len(watched_movie_ids), n_item_features))
    for i, movie_id in enumerate(watched_movie_ids):
        items_info[i] = item_data.iloc[movie_id].to_numpy()

    user_ids = torch.tensor(user_ids).to(dtype=torch.long)
    item_ids = torch.tensor(watched_movie_ids).to(dtype=torch.long)
    users_info = torch.tensor(users_info).to(dtype=torch.float)
    items_info = torch.tensor(items_info).to(dtype=torch.float)

    with torch.no_grad():
        output = model(user_ids=user_ids, item_ids=item_ids,
                       users_info=users_info, items_info=items_info)[0]
        
    watched_movies_with_rating = list(
        zip(watched_movie_ids, output.numpy()))
    
    item_ratings = sorted(watched_movies_with_rating,key=lambda x: x[1])

    return item_ratings

def mean_avg_precision(search_results, relevance):
    # calculate MAP score for search results, treating relevance judgments as binary - either relevant or not.
    #
    # search_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
    # note that for tests to pass, the i-th result in search_results should correspond to (i+1)-th query_id.  
    # relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]       
    def precision_at_k(result, relevant_docs):
        relevant_docs = set(relevant_docs)
        res_set = set(result)
        
        inter = set.intersection(relevant_docs, res_set)
        
        num_rel = len(inter)
        
        return num_rel/len(result)
    
    
    def average_precision_at_k(result, relevant_docs):
        result_list = []
        precision = 0
        for result_doc in result:
            result_list.append(result_doc)
            p_at_k = precision_at_k(result_list, relevant_docs)
            precision += p_at_k * int(result_doc in relevant_docs)
            
        return precision/len(relevant_docs)
        
    map_res = 0
    for i, res in enumerate(search_results):
        relevant_docs = [r[0] for r in relevance[i+1]]
        map_res += average_precision_at_k(res, relevant_docs)
        
    return map_res/len(search_results)


import math 

def NDCG(search_results, relevance):
    # TODO: compute NDCG score for search results. Here relevance is not considered as binary - the bigger
    # the judgement score is, the more relevant is the document to a query. Because in our cranfield dataset relevance
    # judgements are presented in a different way (1 is most relevant, 4 is least), we will need to do smth with it. 

    
    def DCG(result, relevance_dict):
        s = 0
        for i, doc_id in enumerate(result):
            if doc_id in relevance_dict.keys():
                CG = 2**relevance_dict[doc_id]-1
                s += CG/np.log2(i+2)
        return s
    
    def iDCG(relevance_dict):
        s = 0
        for i, relevance in enumerate(sorted(list(relevance_dict.values()), reverse=True)):
            CG = 2**relevance-1
            s += CG/np.log2(i+2)
        return s
    
    s = 0
    count = 0
    for i, result in enumerate(search_results):
        relevance_dict = {item[0]: 5-item[1] for item in relevance[i+1]}
        iDCG_value = iDCG(relevance_dict)
        if iDCG_value != 0.0:
            nDCG = DCG(result, relevance_dict) / iDCG_value
            s += nDCG
        count+=1

    return s/count

def get_relevance_and_results(model, test_data, item_data, user_data):
    test_relevance = {}
    test_results = []
    
    for i, user_id in enumerate(user_data.index.values):
        actual_rating = get_user_watched_movies(user_id=user_id, test_data=test_data)
        test_relevance[i+1] = actual_rating
        
        recommendations = get_watched_movies_recommendation(model=model, user_id=user_id, test_data=test_data, item_data=item_data, user_data=user_data)
        test_results.append([rec[0] for rec in recommendations])
    
    return test_relevance, test_results


if __name__ == '__main__':
    config = Config()
    set_seed(config.seed)

    test_data = pd.read_csv(os.path.join(TEST_DATA_DIR, 'data_test.csv'))
    user_data = pd.read_csv(os.path.join(DATA_DIR, 'user.csv'), index_col=0)
    item_data = pd.read_csv(os.path.join(DATA_DIR, 'item.csv'), index_col=0)

    original_user_ids = user_data.original_user_id - user_data.original_user_id.min()
    original_item_ids = item_data.original_item_id - item_data.original_item_id.min()

    user_data = user_data.drop(columns=['original_user_id'])
    item_data = item_data.drop(columns=['original_item_id'])

    n_user_features = user_data.shape[1]
    n_item_features = item_data.shape[1]

    test_dataset = MovieDataset(test_data, user_data, item_data)

    train_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True)

    model = RecSysModel(n_items=len(item_data), n_users=len(user_data), embedding_dim=config.embedding_dim, hidden_size=config.hidden_size,
                        dropout_rate=config.dropout_rate, n_item_features=n_item_features, n_user_features=n_user_features).to(config.device)

    checkpoint = torch.load(CKPT_PATH)

    model.load_state_dict(checkpoint['model_state_dict'])

    test_relevance, test_results = get_relevance_and_results(model=model, test_data=test_data, item_data=item_data, user_data=user_data) 
                
    map_test = mean_avg_precision(test_results, test_relevance)
    print(f"MAP score: {map_test:.4f}")
    ndcg_test = NDCG(test_results, test_relevance)
    print(f"NDCG score: {ndcg_test:.4f}")
