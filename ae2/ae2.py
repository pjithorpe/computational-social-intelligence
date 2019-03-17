import numpy as np
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("data-part-2.csv")

#Evenly spread smiles and frowns (for better folds)
reordered_data_list = []
for i in range(0,26):
	reordered_data_list.append(data.iloc[i,:].values)
	reordered_data_list.append(data.iloc[-(i+1),:].values)

reordered_data = np.vstack(reordered_data_list)

############################################################

def SplitTrainAndTest(features_matrix, classes_vector, k_folds, fold_no):
	vectors_count = features_matrix.shape[0]

	test_length = int(round(vectors_count / k_folds))

	split_start = test_length * fold_no
	split_end = split_start + test_length

	X_train = np.append(features_matrix[:split_start,:], features_matrix[split_end:,:], axis=0)
	y_train = np.append(classes_vector[:split_start], classes_vector[split_end:], axis=0)

	X_test = features_matrix[split_start:split_end,:]
	y_test = classes_vector[split_start:split_end]

	return X_train, y_train, X_test, y_test

def CalculateLogLikelihood(feature_values, means, variances):
	euclid_dist = np.power((feature_values - means),2)
	divisor = variances * 2

	std_dev_term = variances * 2*np.pi
	std_dev_term = np.sqrt(std_dev_term)
	st_dev_term = np.log(std_dev_term)

	log_likelihoods = st_dev_term + (euclid_dist / divisor)

	log_lh = -np.sum(log_likelihoods)

	return log_lh

def MakePredictions(X, y, c1_means, c2_means, c1_variances, c2_variances):
	predictions = []
	for row in range(0, X_test.shape[0]):
		frown_log_lh = CalculateLogLikelihood(X_test[row,:], c1_means, c1_variances)
		smile_log_lh = CalculateLogLikelihood(X_test[row,:], c2_means, c2_variances)
		
		if(frown_log_lh > smile_log_lh):
			predictions.append("frown")
		else:
			predictions.append("smile")

	# Now check to see how many predictions we got wrong
	errors = 0
	for i in range (0, len(predictions)):
		if predictions[i] != y_test[i]:
			errors += 1

	return predictions, errors


# Part 2

X = np.array(reordered_data[:,:-1], dtype="float64")
y = reordered_data[:,-1]
k = 4

best_errors = np.iinfo(np.int32).max
for fold in range(0,k):
	X_train, y_train, X_test, y_test = SplitTrainAndTest(X, y, k_folds=k, fold_no=fold)

	X_frowns = X_train[np.where(y_train == "frown")]
	X_smiles = X_train[np.where(y_train == "smile")]

	frowns_means = np.mean(X_frowns, axis=0)
	frowns_variances = np.var(X_frowns, axis=0)
	smiles_means = np.mean(X_smiles, axis=0)
	smiles_variances = np.var(X_smiles, axis=0)

	predictions, errors = MakePredictions(X_test, y_test, frowns_means, smiles_means, frowns_variances, smiles_variances)

	if(errors < best_errors):
		best_errors = errors
		best_fold = fold
		c0_means = frowns_means
		c0_variances = frowns_variances
		c1_means = smiles_means
		c1_variances = smiles_variances

	print("Fold no. =", fold, "Error =", np.around(errors/len(predictions)*100, 3), "%")

print("\nBest fold =", best_fold)


predictions, errors = MakePredictions(X, y, c0_means, c1_means, c0_variances, c1_variances)

print("Test Error =", np.around(errors/len(predictions)*100, 3), "%")