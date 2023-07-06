"""
    Script containing definitions of designed classifiers
        - for more info on individual classifiers, see the attached report

    B(E)3M33UI - First semestral task
    Dominik Fischer, 2021
"""
# imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

def tune_MLP_classifier(scorer):
    """ Function that tunes and returns MLP classifier
        :param scorer: custom scoring function
        :return clf: trained pipeline consisting of scaling, upsampling and mlp classification
    """
    # create MLPClassifier model 
    mlp_clf = MLPClassifier(solver='adam', hidden_layer_sizes=(3,32),max_iter=5000,
                            alpha = 0.0001, learning_rate='adaptive')
    # construct pipeline with scaling, upsampling and classification
    clf = make_pipeline(StandardScaler(), SMOTE(), mlp_clf)
    return clf

def tune_svm_classifier(scorer):
    """" Function that tunes and returns the support vector machine classifier 
        :param scorer: custom scoring function
        :return clf: trained pipeline consisting of scaling, upsampling and svm classification
    """
    # create SVC  model - svm classifier
    svm_clf = SVC(kernel = 'rbf')
    # construct pipeline with scaling, upsampling and classification
    clf = make_pipeline(StandardScaler(), SMOTE(), svm_clf)
    # parameters for grid search
    params_selection = {'svc__gamma' : [1e-3,1e-2,1e-1], 'svc__C' : [0.1, 0.2, 0.3] }
    # perform grid search
    clf = GridSearchCV(clf, param_grid = params_selection, scoring=scorer)
    return clf 

def tune_random_forest_classifier(scorer):
    """ Function that tunes and returns the random forest classifier 
        :param scorer: custom scoring function
        :return clf: trained pipeline consisting of scaling, upsampling and random forest classification
    """
    # create RandomForestClassifier model
    rf_clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 40)
    # construct pipeline with scaling, upsampling and classification
    clf = make_pipeline(StandardScaler(), SMOTE(), rf_clf)
    # parameters for grid search
    params_selection = {'randomforestclassifier__max_depth': [3, 4, 5, 6, 7]}
    # perform grid search
    clf = GridSearchCV(clf, param_grid = params_selection, cv = 5, scoring = scorer, return_train_score = False)
    return clf

def tune_logistic_regression_classifier(scorer):
    """ Function that tunes and returns a logsitic regression classifier 
        :param scorer: custom scoring function
        :return clf: trained pipeline consisting of scaling, upsampling and random forest classification
    """
    # values of regularization constant C for internal grid search 
    C_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    # create LogisticRegressionCV model
    log_clf = LogisticRegressionCV(Cs = C_vals, max_iter=500, scoring = scorer, solver = 'newton-cg')
    # construct pipeline with scaling, upsampling and logistic regression
    clf = make_pipeline(StandardScaler(), SMOTE(), log_clf)
    return clf