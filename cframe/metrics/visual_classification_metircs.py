def acc(output, target):
	"""Computes the precision@k for the specified values of k"""
	pred = np.argmax(output, axis=1).reshape(target.shape)
	bath_size = pred.shape[0]
	acc = np.sum(pred == target) / float(bath_size)
	return acc

if __name__ == '__main__':
	import numpy as np
	pred = np.random.random((5, 10))
	target = np.random.random((5, 1)).astype(np.int)
	print(acc(pred,target))