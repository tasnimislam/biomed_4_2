import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

def roc_curve_plot_custom(y_test, y_score, class_used, accuracy, run_number, counts):
    fpr, tpr, _ = roc_curve(y_score/2, y_test/2)
    print(fpr, tpr)
    plt.figure(figsize = (20, 20))
    plt.plot(fpr, tpr)
    plt.title(f'Run number {run_number}: ROC curve for {class_used} distribution {counts} for accuracy: {accuracy} \n y_predict:{y_score} \n y_test:{y_test}')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0, )
    plt.ylim(0, )
    plt.savefig(f'Run number {run_number} ROC curve for {class_used} for accuracy {accuracy}.png')

def predict(X_test, y_test, model, class_used, run_number, counts):
    y_predict = model.predict(X_test)
    print("predict, test",  y_predict, y_test)
    accuracy = np.sum(y_predict == y_test) / len(X_test)
    print('Accuracy:', accuracy)
    roc_curve_plot_custom(y_test, y_predict, class_used, accuracy, run_number, counts)
