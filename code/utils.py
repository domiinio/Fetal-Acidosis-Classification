"""
    Utilities for data load and selection

    B(E)3M33UI - Support script for the first semester task

    Jiri Spilka, 2019
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
import matplotlib.pyplot as plt

PH_THR = 7.05

def load_data_binary(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["y"] = (df.pH <= 7.05).astype(int).ravel()
    return df


def load_data_stage_last_k_segments(df: pd.DataFrame, select_stage: int = 0, nr_seg: int = 1) -> pd.DataFrame:
    """Load k last segments from data

    :param df: pandas dataframe
    :param select_stage: 0 - all, 1 - first stage, 2 - second stage
    :param nr_seg: number of last segments to load
    :return:
    """

    if select_stage == 0:
        return df.loc[df.segIndex <= nr_seg, :]

    elif select_stage == 1:
        ind = np.logical_and(df.segStageI_index > 0, df.segStageI_index <= nr_seg)
        return df.loc[ind, :]

    elif select_stage == 2:
        ind = np.logical_and(df.segStageII_index > 0, df.segStageII_index <= nr_seg)
        return df.loc[ind, :]

    else:
        raise Exception(f"Unknown value select_stage={select_stage}")


def df_drop_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(
        columns=[
            "name",
            "pH",
            "year",
            "segStart_samp",
            "segEnd_samp",
            "segIndex",
            "segStageI_index",
            "segStageII_index",
        ]
    )

    # the stage information might be useful
    df = df.drop(columns=["segStage"])

    # other features that are probably not very useful (correlated to the other ones or irrelevant)
    df = df.drop(
        columns=[
            "bslnMean",
            "bslnSD",
            "decDeltaMedian",
            "decDeltaMad",
            "decDtrdPlus",
            "decDtrdMinus",
            "decDtrdMedian",
            "bslnAllBeta0",
            "bslnAllBeta1",
            "MF_hmin_noint",
            "H310",
            "MF_c1",
            "MF_c2",
            "MF_c3",
            "MF_c4",
        ]
    )

    return df

def drop_nan_values(X, y) -> Tuple[np.array, np.array]:
    """Drop rows containing NaN"""
    idxs = ~np.isnan(X).any(axis = 1)
    return X[idxs] , y[idxs]

def get_X_y_from_dataframe(df: pd.DataFrame) -> Tuple[np.array, np.array, List[str]]:
    """Get feature matrix and labels"""
    y = df.y
    df = df.drop(columns=["y"])
    return df.values, y, list(df)


def g_mean(estimator, X, y):
    y_pred = estimator.predict(X)
    return g_mean_score(y, y_pred)


def g_mean_score(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""

    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("The ground truth values and the predictions may contain at most 2 values (classes).")

    tn = cm[0, 0]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]

    se = tp / float(tp + fn)
    sp = tn / float(tn + fp)
    # metrics.
    g = np.sqrt(se * sp)

    return g

def get_all_metrics_as_dict(dataset: str, y_true: np.array, y_pred: np.array):

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    se = tp / float(tp + fn) #sensitivity
    sp = tn / float(tn + fp) #specificity

    g = np.sqrt(se * sp)

    return {"dataset": dataset, "G-mean": g, "SE": se, "SP": sp, "TP": tp, "FP": fp, "TN": tn, "FN": fn}

def print_results(_df: pd.DataFrame):
    print(_df[["dataset", "G-mean", "SE", "SP", "TP", "FN", "TN", "FP"]])



def plot_roc(models, Xtr, ytr, Xtst, ytst, nfeat=np.inf):
    """
    Compute and plot ROC curves for given models and data.

    :param models: A list of dicts describing the models.
                Each dictionary should have the following items:
               'clf' - the classifier
               'descr' - str description of classifier
               'color' -  color used for plotting
    :param Xtr: training features
    :param ytr: training labels
    :param Xtst: test features
    :param ytst: test labels
    :param nfeat: number of features to be used in the model
    :return:
    """

    nfeat = Xtr.shape[1] if nfeat > Xtr.shape[1] else nfeat
    Xtr = Xtr[:, 0:nfeat]
    Xtst = Xtst[:, 0:nfeat]

    for m in models:
        keys = m.keys()
        assert 'clf' in keys
        assert 'descr' in keys
        assert 'color' in keys  

    for model in models:
        # Process the model information
        clf = model['clf']
        descr = model['descr']
        color = model['color']
        # Fit the classifier to training data
        clf.fit(Xtr, ytr)
        # Get the ROC curve for the classifier
        fpr, tpr, thresholds = roc_curve(ytst, clf.predict_proba(Xtst)[:, 1])
        # Get the AUC
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        legstr = '{:s}, AUC = {:.2f}'.format(descr, roc_auc)
        plt.plot(fpr, tpr, color, label=legstr, lw=2)

        print('## model: ', clf.__repr__(), '\n', 'AUC =', roc_auc)

    # Decorate the graph
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
