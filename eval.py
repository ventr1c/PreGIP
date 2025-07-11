import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

import torch
from GCL.eval import get_split, SVMEvaluator

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def linear_evaluation(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X[idx_train], Y[idx_train])

    y_pred = clf.predict_proba(X[idx_test])
    y_pred = prob_to_one_hot(y_pred)
    y_test = Y[idx_test]

    # micro = f1_score(y_test, y_pred, average="micro")
    # macro = f1_score(y_test, y_pred, average="macro")

    # print(y_pred.argmax(1).shape,y_test.argmax(1))
    # return {
    #     'F1Mi': micro,
    #     'F1Ma': macro
    # }
    # prediction results of idx_test
    prediction = y_pred.argmax(1)
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc, prediction

def train_dt_classifier(embeddings, y, idx_train,):
    # idx_train = idx_train.cpu().numpy()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X[idx_train], Y[idx_train])

    return clf

    # micro = f1_score(y_test, y_pred, average="micro")
    # macro = f1_score(y_test, y_pred, average="macro")

    # print(y_pred.argmax(1).shape,y_test.argmax(1))
    # return {
    #     'F1Mi': micro,
    #     'F1Ma': macro
    # }
    # prediction results of idx_test
    prediction = y_pred.argmax(1)
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc, prediction

def global_test(encoder_model, dataloader, split):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    # split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result