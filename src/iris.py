import pandas as pd
import numpy as np
from libsvm.svmutil import *
from svm_parser import *
from utils.preprocessing import *
import time

cv = 0.1

def main():
    filename = "iris/Iris.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    df = df.head(100)
    # print(df)

    df, x, y = split_features_and_classes(
        df, 'Species', encode=['Species'], drop=None
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


if __name__ == '__main__':
    main()