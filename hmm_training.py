import numpy as np
from hmmlearn import hmm
from csv import reader

def hmm_eat_train(filename_eat_train, hmm_type):
    data_train = []

    if hmm_type == "C":
        model = hmm.CategoricalHMM(n_components=3, n_iter=100)
    elif hmm_type == "P":
         model = hmm.PoissonHMM(n_components=3, n_iter=100)
    elif hmm_type == "G":
         model = hmm.GaussianHMM(n_components=3, n_iter=100)

    with open(filename_eat_train + ".csv", 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            while("" in row):
                row.remove("")
            row = [int(x) for x in row]
            data_train.append(row)

    print(model.transmat_prior)
    print(model.startprob_prior)

    model.fit(data_train)
    score = model.score(data_train)

    aic = model.aic(data_train)
    bic = model.bic(data_train)
    ll = score

    print(f"AIC: {aic} | BIC: {bic} | LL: {ll}")
    print(model.transmat_)
    print(model.startprob_)

def hmm_drink_train(filename_drink_train, hmm_type):
    data_train = []

    if hmm_type == "C":
        model = hmm.CategoricalHMM(n_components=2, n_iter=100)
    elif hmm_type == "P":
         model = hmm.PoissonHMM(n_components=2, n_iter=100)
    elif hmm_type == "G":
         model = hmm.GaussianHMM(n_components=2, n_iter=100)
    elif hmm_type == "M":
        model = hmm.MultinomialHMM(n_components=2, n_iter=100)

    with open(filename_drink_train + ".csv", 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            while("" in row):
                row.remove("")
            row = [int(x) for x in row]
            data_train.append(row)

    print(model.transmat_prior)
    print(model.startprob_prior)

    model.fit(data_train)
    score = model.score(data_train)

    aic = model.aic(data_train)
    bic = model.bic(data_train)
    ll = score

    print(f"AIC: {aic} | BIC: {bic} | LL: {ll}")
    print(model.transmat_)
    print(model.startprob_)