import numpy as np
from hmmlearn import hmm
from csv import reader
import pickle

def hmm_eat_train(filename_eat_train, model_filename):
    data_train = []
    with open(model_filename, "rb") as file: 
      model = pickle.load(file)

    with open(filename_eat_train + ".csv", 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            while("" in row):
                row.remove("")
            row = [int(x) for x in row]
            data_train.append(row)

    print(model.transmat_prior)
    print(model.startprob_prior)

    score = model.score(data_train)

    aic = model.aic(data_train)
    bic = model.bic(data_train)
    ll = score

    print(f"AIC: {aic} | BIC: {bic} | LL: {ll}")
    print(model.transmat_)
    print(model.startprob_)


def hmm_drink_train(filename_drink_train, model_filename):
    data_train = []
    with open(model_filename, "rb") as file: 
      model = pickle.load(file)

    with open(filename_drink_train + ".csv", 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            while("" in row):
                row.remove("")
            row = [int(x) for x in row]
            data_train.append(row)

    print(model.transmat_prior)
    print(model.startprob_prior)

    score = model.score(data_train)

    aic = model.aic(data_train)
    bic = model.bic(data_train)
    ll = score

    print(f"AIC: {aic} | BIC: {bic} | LL: {ll}")
    print(model.transmat_)
    print(model.startprob_)

def hmm_test(test_array, model_filename):
    with open(model_filename, "rb") as file: 
      model = pickle.load(file)

    print(f"Next state: {model.predict(test_array)}")