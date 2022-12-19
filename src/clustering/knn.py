import sys
sys.path.append('./src/')

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from utils.neighbors_centroids import *
from utils.preprocessing import *

train_fraction = 0.9

def main():
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    df, x, y = split_features_and_classes(
        df, 'smoking', encode=['gender', 'tartar'], drop=['ID', 'oral']
    )
    x_train, y_train, x_test, y_test = shuffle_and_keep_for_validation(x, y, 1-train_fraction)

    knn1_start = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    knn1.fit(x_train, y_train)
    knn1_traintime = time.time() - knn1_start
    print(f"  >> Fit model for number of neighbors : 1 [{knn1_traintime:.4f} sec.]")

    knn3_start = time.time()
    knn3 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    knn3.fit(x_train, y_train)
    knn3_traintime = time.time() - knn3_start
    print(f"  >> Fit model for number of neighbors : 3 [{knn3_traintime:.4f} sec.]")
    print()

    test_model(knn1, x_test, y_test, 1)
    test_model(knn3, x_test, y_test, 3)


if __name__ == '__main__':
    main()