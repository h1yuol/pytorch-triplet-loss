import torch

def pairwise_distances(embeddings, squared=False):
	"""
	||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
	"""
	# get dot product (batch_size, batch_size)
	dot_product = embeddings.mm(embeddings.t())

	# a vector
	square_sum = dot_product.diag()

	distances = square_sum.unsqueeze(1) - 2*dot_product + square_sum.unsqueeze(0)

	distances = distances.clamp(min=0)

	if not squared:
		epsilon=1e-16
		mask = torch.eq(distances, 0).float()
		distances += mask * epsilon
		distances = torch.sqrt(distances)
		distances *= (1-mask)

	return distances

# def batch_all_triplet_loss