from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
import nltk

"""
This file contains functions relating to creating files, finding unique words and labels and most importantly
tokenizing data.

The most important functions are:
create_trainable_data_final
create_trainable_data_final2
vectorising2

As these are used in the final implementation.
The other functions here are alternatives or previous versions of functions that could be applied 
for alternative approaches.
"""


# The following function was used to retrieve all unique words form data
def vectorising(data):
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(data)
    return vectorizer.get_feature_names()


# The following function improves upon the first vectoriser by only returning the n most frequent terms to reduce size
def vectorising2(data, amount_of_words):
    vectorizer = CountVectorizer().fit(data)
    bow = vectorizer.transform(data)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:amount_of_words]


# Tokenizes the training features and returns the tokenized data
def create_trainable_data_final(in_data, unique_items):
    # Create a list with the amount of unique words as features and the length of the dataset
    train_data = np.array([[0] * len(unique_items) for _ in range(len(in_data))])
    # Keeps track of which row of training data currently on to corretly manipulate list
    row = 0
    # Iterate through all the data
    for x in in_data:
        # Iterate through each word of each row
        for i in x[0].split():
            # If the word is in the list when find what index and change that index in the tokenized data from 0 to 1
            word = i.lower()
            if word in unique_items:
                train_data[row][np.where(unique_items == word)[0][0]] = 1
        row += 1
    return train_data


# Tokenizes the training labels and returns the tokenized data
def create_trainable_data_final2(in_data, unique_items):
    # Create a list with the amount of unique words as features and the length of the dataset
    train_data = np.array([[0] * len(unique_items) for _ in range(len(in_data))])
    # Keeps track of which row of training data currently on to corretly manipulate list
    row = 0
    # Iterate through all the data
    for x in in_data:
        # Iterate through each word of each row
        for i in x.split():
            # If the word is in the list when find what index and change that index in the tokenized data from 0 to 1
            if i in unique_items:
                train_data[row][np.where(unique_items == i)[0][0]] = 1
        row += 1
    return train_data


# Manually iterated through data and writes all the unique words to a file
def create_unique(in_data):
    with open('unique_labels_econbiz.csv', mode='w', newline='', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        unique = np.array([])
        progress = 0
        for x in in_data:
            if progress % 10000 == 0:
                print(progress)
            for i in x.split('\t'):
                if progress == 1:
                    print(i)
                if i not in unique:
                    unique = np.append(unique, i)
                    file_writer.writerow([i])
            progress += 1
    print("done writing")


"""
The first iteration of the function that creates training data.
It first found all the unique words in the features or all unique labels
The first parameter determines if the input is features or labels as these are different formats and labels should
not be stemmed

The data is then tokenized
"""


def find_unique(feat_or_lab, training, testing):
    # Stemmer set to use English
    s = nltk.stem.SnowballStemmer('english')
    # Holds
    unique = np.array([])
    # Goes through training data
    for x in training:
        if feat_or_lab == 1:
            splitted = x.split()
            for i in splitted:
                if i not in unique:
                    unique = np.append(unique, i)
        else:
            splitted = x[0].split()
            for i in splitted:
                word = s.stem(i)
                if word not in unique:
                    unique = np.append(unique, word)
    # Goes through testing data
    for x in testing:
        if feat_or_lab == 1:
            splitted = x.split()
            for i in splitted:
                if i not in unique:
                    unique = np.append(unique, i)
        else:
            splitted = x[0].split()
            for i in splitted:
                word = s.stem(i)
                if word not in unique:
                    unique = np.append(unique, word)

    train_data = create_trainable_data(feat_or_lab, training, unique)
    test_data = create_trainable_data(feat_or_lab, testing, unique)
    return train_data, test_data


# First version of okenizing the data manually. Using stemmer
def create_trainable_data(feat_or_lab, in_data, unique_items):
    s = nltk.stem.SnowballStemmer('english')
    train_data = np.array([[0] * len(unique_items) for _ in range(len(in_data))])
    row = 0
    for x in in_data:
        if feat_or_lab == 1:
            splitted = x.split()
            for i in splitted:
                train_data[row][np.where(unique_items == i)[0][0]] = 1
        else:
            splitted = x[0].split()
            for i in splitted:
                train_data[row][np.where(unique_items == s.stem(i))[0][0]] = 1
        row += 1
    return train_data


# Alternative to CountVectorizer that find all unique words. Much slower in practice
def find_unique2(training):
    s = nltk.stem.SnowballStemmer('english')
    unique = np.array([])
    round = 0
    for x in training:
        if round % 10000 == 0:
            print(round)
        for i in x[0].split():
            word = s.stem(i)
            if word not in unique:
                unique = np.append(unique, word)
        round += 1

    create_trainable_data3(training, unique)


"""
Manually tokenizes data and writes it to a file
Ended up not being used for features and PubMed as the filesize would be too large
"""


def create_trainable_data3(in_data, unique_items):
    s = nltk.stem.SnowballStemmer('english')
    with open('vectorised_features.csv', mode='w', newline='', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        progress = 0
        for x in in_data:
            if progress % 100 == 0:
                print("Progress:", progress)
            train_data = np.array([0 for _ in range(len(unique_items))])
            for i in x[0].split():
                train_data[np.where(unique_items == s.stem(i))[0][0]] = 1
            file_writer.writerow(train_data)
            progress += 1
    print("done writing")


# Creates a csv file of any data
def create_csv(fname, data):
    with open(fname + '.csv', mode='w', newline='', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        for d in zip(data):
            file_writer.writerow(d)
