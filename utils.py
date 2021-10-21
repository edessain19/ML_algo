import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None, df_option=True):
	result = []
	labels_hat = np.sort(np.unique(y_hat))
	labels_y = np.sort(np.unique(y_true))
	if labels != None:
		labels_y = labels
	elif len(labels_hat) > len(labels_y):
		labels_y = labels_hat
	for row in labels_y:
		row_ = []
		for col in labels_y:
			nb = 0
			for yi, ypi in zip(y_true, y_hat):
				if yi == row and ypi == col:
					nb += 1
			row_.append(nb)
		result.append(row_)
	result = np.array(result)
	if df_option == True:
		data = {}
		i = 0
		for label in labels_y:
			data[label] = result[:, i]
			i += 1
		result = pd.DataFrame(data, labels_y, labels_y)
	return result

def accuracy_score_(y, y_hat):
	p = 0
	for yi, ypi in zip(y, y_hat):
		if yi == ypi:
			p += 1
	return (p) / len(y)

def precision_score_(y, y_hat, pos_label=1):
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	for yi, ypi in zip(y, y_hat):
		if yi == ypi and yi == pos_label:
			tp += 1
		elif yi == ypi and yi != pos_label:
			tn += 1
		elif yi != ypi and yi == pos_label:
			fn += 1
		else:
			fp += 1
	return tp / (tp + fp)

def recall_score_(y, y_hat, pos_label=1):
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	for yi, ypi in zip(y, y_hat):
		if yi == ypi and yi == pos_label:
			tp += 1
		elif yi == ypi and yi != pos_label:
			tn += 1
		elif yi != ypi and yi == pos_label:
			fn += 1
		else:
			fp += 1
	return tp / (tp + fn)

def f1_score_(y, y_hat, pos_label=1):
	prec = precision_score_(y, y_hat, pos_label)
	rec = recall_score_(y, y_hat, pos_label)
	return (2 * prec * rec)/(prec + rec)