import numpy as np
import torch 
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader
import random



def distribute_data_iid(
    dataset, num_clients : int, 
    seed : int, 
    shuffle: bool = True
    ) -> dict:
    """"
    Inputs : 
        dataset : Pytorch dataset objecct, which has to be distributed
        num_clients : The number of clients among which the data shall be distributed 
        seed : random number generator seed

        shufle : Bool to activate/deactivate shuffling

     
    Output : 
        dataset : dictionary with client ID as keys and corresponding dataset as values.
    """

    np.random.seed(seed)
    num_classes = len(dataset.classes)

    # creating a map to get the indices of each class
    class_to_idx = {i:[] for i in range(num_classes)}

    for idx, label in enumerate(dataset.targets) : class_to_idx[label].append(idx)


    # distributing data to each client
    client_ind_map = {i : [] for i in range(num_clients)}

    for label in range(num_classes): 
        n_samples = len(class_to_idx[label])

        indices = random.sample(class_to_idx[label], n_samples) if shuffle else class_to_idx[label]
        label_dist = [n_samples//num_clients for i in range(num_clients)]

        start_idx = 0
        for client_no, n_data in enumerate(label_dist):
            client_ind_map[client_no] += indices[start_idx : start_idx + n_data]
            start_idx += n_data

    datasets = { client_no : Subset(dataset, ind ) for client_no, ind in zip (client_ind_map.keys(), client_ind_map.values())}

    return datasets





def distribute_data_dirichilet(
    dataset, num_clients : int, 
    alphas : list[int], 
    seed : int, 
    shuffle : bool = True
) -> dict:
    """"
    Inputs : 
        dataset : Pytorch dataset objecct, which has to be distributed
        num_clients : The number of clients among which the data shall be distributed 
        alphas : List containing hyperparameters of the dirichilet dirtsibution of length num_clients.
        seed : random number generator seed

        shufle : Bool to activate/deactivate shuffling

     
    Output : 
        dataset : dictionary with client ID as keys and corresponding dataset as values.
    """
    
    np.random.seed(seed)
    num_classes = len(dataset.classes)

    dirichlet = np.random.dirichlet(alpha=alphas, size=num_classes)

    # creating a map to get the indices of each class
    class_to_idx = {i:[] for i in range(num_classes)}

    for idx, label in enumerate(dataset.targets) : class_to_idx[label].append(idx)


    # distributing data to each client
    client_ind_map = {i : [] for i in range(num_clients)}

    for label in range(num_classes): 
        n_samples = len(class_to_idx[label])

        indices = random.sample(class_to_idx[label], n_samples) if shuffle else class_to_idx[label]
        # Sampling from distribution
        label_dist = np.floor(n_samples*dirichlet[label]).astype('int')

        start_idx = 0
        for client_no, n_data in enumerate(label_dist):
            client_ind_map[client_no] += indices[start_idx : start_idx + n_data]
            start_idx += n_data

    datasets = { client_no : Subset(dataset, ind ) for client_no, ind in zip (client_ind_map.keys(), client_ind_map.values())}

    return datasets
