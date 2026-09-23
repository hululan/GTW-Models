import numpy as np
import pandas as pd
from scipy.stats import rankdata
kernel_list = ['bisquare','gaussian','inverse','normal','NN','exponential']
def kernel_function(
        distances:np.ndarray,
        bandwidth:float|int,
        kernel:str,
        pow:int=1,
)->np.ndarray:
    """
    Calculating the weight matrix by using kernel function,
    kernel types : 'bisquare','gaussian','inverse','normal','NN': nearest neighbours
    :param distances: np.array the (m*n) distance matrix between each pair of points
    :param bandwidth: float the bandwidth of the kernel, or int indicating the i neighbours of the observation
    :param kernel: str the kernel type
    :param pow: int, the power of the kernel
    :return: weights matrix
    indicating the weight between each pair of points, the i row j column showing the weight observation j of location i,
    """

    normalized_distances = abs(distances / bandwidth)

    if kernel == 'normal':
        weights = np.ones_like(distances, dtype = int)
    elif kernel == 'gaussian':
        weights = np.exp(-np.power(normalized_distances,2))
    elif kernel =='exponential':
        normalized_distances +=(normalized_distances.min()*0.001)# avoid divide by zero
        weights = np.exp(-normalized_distances)
    elif kernel == 'inverse':
        normalized_distances +=(normalized_distances.min()*0.001)# avoid divide by zero
        weights = np.power(normalized_distances,-pow)
    elif kernel == 'bisquare':
        weights = np.power(1-np.power(normalized_distances,2),2)
    elif kernel == 'NN':
        bandwidth = min(bandwidth,(distances.shape[1]-1))
        selected = np.partition(distances,int(bandwidth),axis=1)[:,bandwidth:bandwidth+1]
        weights = np.power(1-np.power(distances/selected,pow),pow)
    else:
        raise ValueError(f'kernel must be one of {kernel_list}')
    return weights


def find_neighbours(
        distances:np.ndarray,
        bandwidth:float|int,
        kernel:str,
):
    """

    :param distances: distance matrix between each pair of points
    :param bandwidth: a float or integer indicating the bandwidth
    :param kernel: kernel type
    :return:
    A boolean matrix showing which observation j is the neighbour of the observation i
    """
    if kernel not in kernel_list:
        raise ValueError(f'kernel must be one of {kernel_list}')
    elif kernel =='NN':
        selected = rankdata(distances,axis=1 ,method='ordinal')
        neighbours = selected<=bandwidth
    else:
        neighbours  = distances<bandwidth
    return neighbours


def adaptive_bandwidth(
        distances:np.ndarray,
        bandwidth:int,
):
    row_sort = distances.argsort(axis=1)
    return row_sort<bandwidth


def fixed_bandwidth(
        distances:np.ndarray,
        bandwidth:float,
):
    """

    :param distances: distance matrix between each pair of points
    :param bandwidth: bandwidth within which will be selected
    :return:
        A boolean matrix showing which observation j is the neighbour of the observation i
    """
    return distances<=bandwidth


