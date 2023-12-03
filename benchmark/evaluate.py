import math
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATA_DIR = 'data/internal'
TEST_DATA_DIR = 'benchmark/data'
CKPT_PATH = 'models/embedding-based/best.pt'


class Config:
    """
    Configuration class for the recommendation system model.

    Attributes:
    - device (str): Device to be used for training ('cpu' or 'cuda').
    - epochs (int): Number of training epochs.
    - seed (int): Random seed for reproducibility.
    - batch_size (int): Batch size for training.
    - embedding_dim (int): Dimensionality of the user and item embeddings.
    - hidden_size (int): Size of the hidden layers in the fully connected network.
    - dropout_rate (float): Dropout rate to prevent overfitting.
    - lr (float): Learning rate for the optimizer.
    """
    device = 'cpu'
    epochs = 20
    seed = 0
    batch_size = 128
    embedding_dim = 32
    hidden_size = 16
    dropout_rate = 0.5
    lr = 1e-3


def set_seed(seed_value=0):
    """
    Set random seeds for reproducibility in Python and PyTorch.

    Args:
    - seed_value (int): Seed value to use for random number generation.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class RecSysModel(nn.Module):
    """
    A PyTorch-based neural network model for collaborative filtering in recommendation systems.

    Args:
    - n_users (int): Number of users in the recommendation system.
    - n_items (int): Number of items in the recommendation system.
    - embedding_dim (int): Dimensionality of the user and item embeddings.
    - hidden_size (int): Size of the hidden layers in the fully connected network.
    - dropout_rate (float): Dropout rate to prevent overfitting.
    - n_item_features (int): Number of features for each item in the recommendation system.
    - n_user_features (int): Number of features for each user in the recommendation system.
    """
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
        """
        Forward pass of the model.

        Args:
        - user_ids (torch.Tensor): Tensor containing user IDs.
        - item_ids (torch.Tensor): Tensor containing item IDs.
        - users_info (torch.Tensor): Tensor containing additional user features.
        - items_info (torch.Tensor): Tensor containing additional item features.

        Returns:
        - torch.Tensor: Predicted ratings for the given user-item pairs.
        """
        user_embeds = self.user_embed(user_ids)
        item_embeds = self.item_embed(item_ids)

        x = torch.cat([user_embeds, item_embeds,
                      users_info, items_info], dim=1)

        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


def dup_rows(a, indx, num_dups=1):
    """
    Duplicate a specific row in a numpy array along a specified axis.

    Args:
    - a (numpy.ndarray): The input array.
    - indx (int): The index of the row to be duplicated.
    - num_dups (int, optional): The number of times to duplicate the row. Default is 1.

    Returns:
    - numpy.ndarray: An array with the specified row duplicated.
    """
    return np.insert(a, [indx+1]*num_dups, a[indx], axis=0)


def get_user_watched_movies(user_id, test_data):
    """
    Retrieves the movies watched by a specific user along with their ratings.

    Args:
    - user_id (int): The ID of the user for whom watched movies are to be retrieved.
    - test_data (pd.DataFrame): DataFrame containing test data.

    Returns:
    - list: A list of tuples, where each tuple contains the item ID and rating of a watched movie by the user.
    """
    watched_movies = test_data[test_data.user_id == user_id].item_id
    watched_movie_ids = watched_movies.to_list()

    watched_movies_with_rating = []

    for i, movie_id in enumerate(watched_movie_ids):
        rating = test_data[(test_data.item_id == movie_id) & (
            test_data.user_id == user_id)].rating.values[0]
        watched_movies_with_rating.append((movie_id, rating))

    item_ratings = sorted(watched_movies_with_rating, key=lambda x: x[1])

    return item_ratings


def get_watched_movies_recommendation(model, user_id, test_data, item_data, user_data):
    model.eval()

    watched_movies = test_data[test_data.user_id == user_id].item_id
    watched_movie_ids = watched_movies.to_list()

    users_info_one_row = user_data.iloc[user_id]

    users_info = dup_rows(users_info_one_row.to_numpy()[np.newaxis, ...],
                          0, len(watched_movie_ids)-1)

    user_ids = [user_id] * len(watched_movie_ids)

    items_info = np.zeros((len(watched_movie_ids), item_data.shape[1]))
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

    item_ratings = sorted(watched_movies_with_rating, key=lambda x: x[1])

    return item_ratings


def mean_avg_precision(search_results, relevance):
    """
    Computes the Mean Average Precision (MAP) for a set of search results.

    Args:
    - search_results (list): List of lists, where each inner list represents the ranked search results.
    - relevance (dict): Dictionary mapping user indices to a list of (item_id, relevance_score) tuples.

    Returns:
    - float: The Mean Average Precision score across all users.
    """
    def precision_at_k(result, relevant_docs):
        """
        Computes Precision at k for a given result.

        Args:
        - result (list): List of item IDs representing the ranked search result.
        - relevant_docs (list): List of relevant item IDs.

        Returns:
        - float: Precision at k.
        """
        relevant_docs = set(relevant_docs)
        res_set = set(result)

        inter = set.intersection(relevant_docs, res_set)

        num_rel = len(inter)

        return num_rel/len(result)

    def average_precision_at_k(result, relevant_docs):
        """
        Computes Average Precision at k for a given result.

        Args:
        - result (list): List of item IDs representing the ranked search result.
        - relevant_docs (list): List of relevant item IDs.

        Returns:
        - float: Average Precision at k.
        """
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


def NDCG(search_results, relevance):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) for a set of search results.

    Args:
    - search_results (list): List of lists, where each inner list represents the ranked search results.
    - relevance (dict): Dictionary mapping user indices to a list of (item_id, relevance_score) tuples.

    Returns:
    - float: The average NDCG score across all users.
    """
    def DCG(result, relevance_dict):
        """
        Computes the Discounted Cumulative Gain (DCG) for a given result.

        Args:
        - result (list): List of item IDs representing the ranked search result.
        - relevance_dict (dict): Dictionary mapping item IDs to their relevance scores.

        Returns:
        - float: The DCG score for the given result.
        """
        s = 0
        for i, doc_id in enumerate(result):
            if doc_id in relevance_dict.keys():
                CG = 2**relevance_dict[doc_id]-1
                s += CG/np.log2(i+2)
        return s

    def iDCG(relevance_dict):
        """
        Computes the Ideal Discounted Cumulative Gain (iDCG) for a given set of relevance scores.

        Args:
        - relevance_dict (dict): Dictionary mapping item IDs to their relevance scores.

        Returns:
        - float: The iDCG score for the given relevance scores.
        """
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
            count += 1

    return s/count


def get_relevance_and_results(model, test_data, item_data, user_data):
    """
    Generates relevance information and recommendation results for a recommendation model.

    Args:
    - model: The recommendation system model.
    - test_data (pd.DataFrame): DataFrame containing test data.
    - item_data (pd.DataFrame): DataFrame containing item data.
    - user_data (pd.DataFrame): DataFrame containing user data.

    Returns:
    - test_relevance (dict): Dictionary mapping user indices to their actual watched movie ratings.
    - test_results (list): List of lists containing recommended movie indices for each user.
    """
    test_relevance = {}
    test_results = []

    for i, user_id in enumerate(user_data.index.values):
        actual_rating = get_user_watched_movies(
            user_id=user_id, test_data=test_data)
        test_relevance[i+1] = actual_rating

        recommendations = get_watched_movies_recommendation(
            model=model, user_id=user_id, test_data=test_data, item_data=item_data, user_data=user_data)
        test_results.append([rec[0] for rec in recommendations])

    return test_relevance, test_results


def get_model(config, n_items, n_users, n_item_features, n_user_features):
    """
    Loads a pre-trained recommendation system model based on the provided configuration and data parameters.

    Args:
    - config (Config): An object containing configuration parameters for the model.
    - n_items (int): Number of items in the recommendation system.
    - n_users (int): Number of users in the recommendation system.
    - n_item_features (int): Number of features for each item in the recommendation system.
    - n_user_features (int): Number of features for each user in the recommendation system.

    Returns:
    - model (RecSysModel): A pre-trained recommendation system model loaded with weights from a checkpoint.
    """
    model = RecSysModel(n_items=len(item_data), n_users=len(user_data), embedding_dim=config.embedding_dim, hidden_size=config.hidden_size,
                        dropout_rate=config.dropout_rate, n_item_features=n_item_features, n_user_features=n_user_features).to(config.device)

    checkpoint = torch.load(CKPT_PATH)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def get_data():
    """
    Reads and preprocesses data from CSV files and returns them as pandas DataFrames.

    Returns:
    - test_data (pd.DataFrame): DataFrame containing test data from 'data_test.csv'.
    - user_data (pd.DataFrame): DataFrame containing user data from 'user.csv' with 'original_user_id' column dropped.
    - item_data (pd.DataFrame): DataFrame containing item data from 'item.csv' with 'original_item_id' column dropped.
    """
    test_data = pd.read_csv(os.path.join(TEST_DATA_DIR, 'data_test.csv'))
    user_data = pd.read_csv(os.path.join(DATA_DIR, 'user.csv'), index_col=0)
    item_data = pd.read_csv(os.path.join(DATA_DIR, 'item.csv'), index_col=0)

    user_data = user_data.drop(columns=['original_user_id'])
    item_data = item_data.drop(columns=['original_item_id'])

    return test_data, user_data, item_data


if __name__ == '__main__':
    config = Config()
    set_seed(config.seed)

    test_data, user_data, item_data = get_data()

    model = get_model(config=config, n_items=len(item_data), n_users=len(
        user_data), n_item_features=item_data.shape[1], n_user_features=user_data.shape[1])

    test_relevance, test_results = get_relevance_and_results(
        model=model, test_data=test_data, item_data=item_data, user_data=user_data)

    map_test = mean_avg_precision(test_results, test_relevance)
    print(f"MAP score: {map_test:.4f}")
    ndcg_test = NDCG(test_results, test_relevance)
    print(f"NDCG score: {ndcg_test:.4f}")