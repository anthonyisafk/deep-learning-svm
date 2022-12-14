import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestCentroid
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

    train_start = time.time()
    model = NearestCentroid()
    model.fit(x_train, y_train)
    traintime = time.time() - train_start
    print(f"  >> Fit model for Nearest Centroid Classifier : {traintime:.4f} sec.")

    test_model(model, x_test, y_test, 0)


if __name__ == '__main__':
    main()