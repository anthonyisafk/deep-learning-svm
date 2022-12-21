"""
Neural Networks & Deep Learning
Aristotle University Thessaloniki - School of Informatics.
******************************
@brief: Building an SVM with libsvm.
        Trained on the `Body Signal of Smoking` dataset.
        (https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
2022 AUTh Electrical and Computer Engineering.
"""

import pandas as pd
import numpy as np
from libsvm.svmutil import *
from svm_parser import *
from utils.preprocessing import *
import time

cv = 0.1

def main():
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    # print(df)

    # df = df.head(1000)
    df, x, y = split_features_and_classes(
        df, 'smoking', encode=['gender', 'tartar'], drop=['ID', 'oral']
    )
    x_train, y_train, x_cv, y_cv = shuffle_and_keep_for_validation(x, y, cv)

    parser = SVMParser(1 / np.shape(df)[1])
    args = parser.parse_args()
    param_str = get_params_str(parser)
    problem = svm_problem(y_train, x_train, isKernel=False)
    param = svm_parameter(param_str)

    start_time = time.time()
    model = svm_train(problem, param)
    stop_time = time.time() - start_time

    _, acc, _ = svm_predict(y_cv, x_cv, model)
    print(acc)

    print(f"\n\n  >> params : {param_str}\n  >> training time : {stop_time} sec.")
    log_info = get_params_csv(args)
    log_info += f",{stop_time:.3f},{acc[0]:2.3f}"
    log_results(f"logs/smoking-t{args.t}.csv", log_info)

if __name__ == '__main__':
    main()