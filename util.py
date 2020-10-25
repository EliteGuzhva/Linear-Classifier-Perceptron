from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score

def get_best_score(model):
    print(model.best_score_)
    print(model.best_params_)
    print(model.best_estimator_)

    return model.best_score_

def print_validation_report(y_true, y_pred):
    average = 'weighted'

    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc_sc)
    prec_sc = precision_score(y_true, y_pred, average=average)
    print("Precision:", prec_sc)
    f1_sc = f1_score(y_true, y_pred, average=average)
    print("F1:", f1_sc)
    rec_sc = recall_score(y_true, y_pred, average=average)
    print("Recall:", rec_sc)
