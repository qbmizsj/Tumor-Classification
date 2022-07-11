from torch.nn import functional as F
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


def cross_entropy_loss(output, labels):
	#print("output, labels:",output, labels,output.shape, labels.shape)
	#labels = labels - 2*torch.ones_like(labels)
	#preds = output.max(1)[1].type_as(labels)
	return F.cross_entropy(output, labels, size_average=True)

def softmax_mse_loss(input_logits, target_logits):
	assert input_logits.size() == target_logits.size()
	#print("output, labels:", input_logits, target_logits)
	input_softmax = F.softmax(input_logits, dim=1)
	target_softmax = F.softmax(target_logits, dim=1)
	#print("output, labels:", input_softmax, target_softmax, input_softmax.shape, target_softmax.shape)
	loss = torch.nn.MSELoss()

	return loss(input_softmax, target_softmax)

def sensitivity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred)
	# 切片操作，获取每一个类别各自的 tn, fp, tp, fn
	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, -1] # False Positive

	tp_sum = CM[-1, -1] # True Positive
	fn_sum = CM[-1, 0] # False Negative
	# 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
	Condition_negative = tp_sum + fn_sum + 1e-6
	sensitivity = tp_sum / Condition_negative

	return sensitivity

def specificity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred)
	#print("CM:", CM)

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, -1] # False Positive

	tp_sum = CM[-1, -1] # True Positive
	fn_sum = CM[-1, 0] # False Negative

	Condition_negative = tn_sum + fp_sum + 1e-6
	Specificity = tn_sum / Condition_negative

	return Specificity


def embedding_evaluation(model, loader, device):
	model.eval()
	raw, y = model.get_embeddings(loader, device)
	#y = y - 2 * np.ones_like(y)

	acc = accuracy_score(raw, y)
	#f1 = f1_score(raw, y, average='macro')
	f1 = f1_score(raw, y)
	sen_score = sensitivity(raw, y)
	spe_score = specificity(raw, y)

	return acc, f1, sen_score, spe_score

