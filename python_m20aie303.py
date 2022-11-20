# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree


import pdb
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n","--name", required=True,
	help="name of the classifier")
args = vars(ap.parse_args())



from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score
x=args["name"]
print(x)
n_cv = 1
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
        
    )

    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }
    for x in models_of_choice:
        clf = models_of_choice[x]
        print("[{}] Running hyper param tuning for {}".format(n,x))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[x], model_path=None
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not x in results:
            results[x]=[]    

        results[x].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        
        file1 = open(x+".txt","a")
        # \n is placed to indicate EOL (End of Line)
        file1.writelines(f"Classification report for classifier {clf}:\n")
        file1.writelines(f"{metrics.classification_report(y_test, predicted)}\n")
        
        
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
            
            
        )


        file1.close()

#print(results)
print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)
print("x_dev: ",x_dev.shape)
print("y_dev: ",y_dev.shape)





