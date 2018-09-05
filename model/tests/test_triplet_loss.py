import numpy as np 
import torch

from model.triplet_loss import pairwise_distances as pairwise_distances_pytorch
from model.triplet_loss import get_valid_triplets_mask
from model.triplet_loss import batch_all_triplet_loss

def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
            pairwise_distances.diagonal())
    return pairwise_distances

def test_pairwise_distances():
    """Test the pairwise distances function."""
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]  # to get distance 0

    for squared in [True, False]:
    	res_np = pairwise_distance_np(embeddings, squared=squared)
    	res_pytorch = pairwise_distances_pytorch(torch.from_numpy(embeddings), squared=squared)
    	assert np.allclose(res_np, res_pytorch.numpy())

def test_triplet_mask():
    """Test function _get_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)

    mask_pytorch = get_valid_triplets_mask(torch.from_numpy(labels)).numpy()

    assert np.allclose(mask_np, mask_pytorch)

def test_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining"""
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    for squared in [True, False]:
        pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        num_positives = 0.0
        num_valid = 0.0
        for i in range(num_data):
            for j in range(num_data):
                for k in range(num_data):
                    distinct = (i != j and i != k and j != k)
                    valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                    if distinct and valid:
                        num_valid += 1.0

                        pos_distance = pdist_matrix[i][j]
                        neg_distance = pdist_matrix[i][k]

                        loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                        loss_np += loss

                        num_positives += (loss > 0)

        loss_np /= num_positives

        # Compute the loss in TF.
        loss_pytorch, fraction = batch_all_triplet_loss(torch.from_numpy(labels), torch.from_numpy(embeddings), margin, squared=squared)

        assert np.allclose(loss_np, loss_pytorch.item())
        assert np.allclose(num_positives / num_valid, fraction.item())





