#!/usr/bin/python37
import pandas as pd
import numpy as np
import MLP
from sklearn.model_selection import KFold
import csv
import assignment2_utils

if __name__ == '__main__':
    # The section below is used to load a dataset and find all unique features or labels
    """
    data = pd.read_csv("./data/econbiz.csv")
    performance = np.array([])
    labels = data.labels
    np.array(assignment2_utils.create_unique(labels.values.reshape(-1)))
    print("done")
    breakpoint()
    """

    # The section below is used to load a small section of  PubMed data and train on it.
    """
    amount_per_read = 100000
    training_data = pd.read_csv("./data/pubmed.csv", nrows=amount_per_read)
    labels = training_data.labels
    training_data = training_data.drop(["id", "fold", "labels"], axis=1)
    labels = np.array(labels)
    training_data = np.array(training_data)
    unique_words = pd.read_csv("unique_words_pubmed.csv", header=None)
    unique_labels = pd.read_csv("unique_labels_pubmed.csv", header=None)
    print(len(unique_labels))
    performance = np.array([])
    """
    # The section below is used to load a small section of  Econbiz data and then performs 10-fold cross-validation
    """
    amount_per_read = 20000
    training_data = pd.read_csv("./data/econbiz.csv", nrows=amount_per_read)
    training_data = training_data.drop(["id", "fold", "labels"], axis=1)
    labels = pd.read_csv("vectorised_label100.csv", nrows=amount_per_read, header=None)
    labels = np.array(labels)
    training_data = np.array(training_data)
    unique_words = np.array(pd.read_csv("unique_words_econbiz3.csv", header=None))
    unique_words = np.array(unique_words)

    performance = np.array([])

    skf = KFold(n_splits=10)

    iteration = 0
    for train, test in skf.split(training_data, labels):
        print(iteration, "/ 10 iterations")
        training_features = training_data[train]
        testing_features = training_data[test]
        training_labels = labels[train]
        testing_labels = labels[test]

        training_features = assignment2_utils.create_trainable_data_final(training_features.reshape(-1), unique_words)
        #training_labels = assignment2_utils.create_trainable_data_final2(training_labels.reshape(-1), unique_labels)

        MLP.mlp_process(iteration, training_features, training_labels, training_features.shape[1],
                        training_labels.shape[1])
        del training_features
        del training_labels
        testing_features = assignment2_utils.create_trainable_data_final(testing_features.reshape(-1), unique_words)
        #testing_labels = assignment2_utils.create_trainable_data_final2(testing_labels.reshape(-1), unique_labels)
        iteration += 1
        performance = np.append(performance, MLP.eval_performance("mynet{}".format(iteration) + ".pt", testing_features,
                                                                  testing_labels, testing_labels.shape[1]))
        del testing_features
        del testing_labels

    print("F1-Score:", performance.mean(), "+/-", performance.std())
    """

    # The section below performs part-by-part loading of data and then performs 10-fold cross-validation
    performance = []
    # Speify filenames of files to load
    unique_file = "unique_words_econbiz3.csv"
    unique_file = "unique_words_pubmed.csv"
    fname = "./data/econbiz.csv"
    fname = "./data/pubmed.csv"
    # Load list of unique words
    unique_words = pd.read_csv(unique_file)
    unique_words = np.array(unique_words)
    # Load list of unique labels (PubMed only as the EconBiz labels are pre-tokenized)
    unique_labels = pd.read_csv("unique_labels_pubmed.csv")
    unique_labels = np.array(unique_labels)
    # Creates a list of 10*10 representing each permutation of 10-fold cross-validation. Each fold will be used for testing once
    perms = []
    li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(0, 10):
        li.insert(0, li.pop())
        perms.append(li.copy())

    # Drop the id, label and fold columns
    cols_to_drop = [0, 2, 3]
    cross = 0
    # Iterate over each permutation defined above
    for p in perms:
        i = 0
        print(cross, "out of 10 permutations")
        # Iterate over each number of the permutation using the last one in the list as
        for val in p:
            print("Fold:", i + 1)
            # Specifies amount of row to read each fold
            amount_per_read = 12000
            # Start value, specifies what row to start reading from. Using number from permutation list
            fr = val * amount_per_read
            # Load training data from file
            training_data = pd.read_csv(fname, skiprows=fr, nrows=amount_per_read)
            # Load labels (PubMed only) line commented out when EconBiz used
            labels = np.array(training_data.iloc[:, 2])
            # Drop id, labels and folds from features
            training_data = training_data.drop(training_data.columns[cols_to_drop], axis=1)

            # Tokenize features and labels (PubMed only)
            training_data = assignment2_utils.create_trainable_data_final(training_data, unique_words)
            labels = assignment2_utils.create_trainable_data_final2(labels, unique_labels)

            # In the case of EconBiz the labels can be read from a file instead of tokenized to save time
            # labels = pd.read_csv("vectorised_label100.csv", skiprows=fr, nrows=amount_per_read)
            # labels = np.array(labels)

            """
            The first fold created the MLP, the final fold evaluates performance.
            All other folds will load a trained model and continue training
            """
            if i == 0:
                MLP.mlp_process(i, training_data, labels, training_data.shape[1], labels.shape[1])
                del training_data
                del labels
            elif i == 9:
                res = MLP.eval_performance("mynet{}".format(i) + ".pt", training_data, labels, labels.shape[1])
                performance = np.append(performance, res)
            else:
                MLP.mlp_train_more(i, training_data, labels)
            i += 1
        cross += 1
    print("F1-Score:", performance, "+/-", performance.std())
