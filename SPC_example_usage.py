from SPC_module.SPC import SPCAlgorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings

from river import compat
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from sklearn import linear_model
from river import tree
from sklearn import datasets
  


def init_estimator_SGDClassifier():
    model = preprocessing.StandardScaler()
    model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(
        loss='log_loss',
        eta0=0.01,
        learning_rate='constant'
    ),
    classes=[False, True]
    )
    return model

def init_estimator_LogisticRegression():
    # ATENÇÃO: ValueError: LogisticRegression(penalty=None, random_state=42) does not have a partial_fit method
    model = preprocessing.StandardScaler()
    model |= compat.convert_sklearn_to_river(
    estimator=linear_model.LogisticRegression(
        penalty = None,
        random_state = 42
    ),
    classes=[False, True]
    )
    return model

def init_estimator_DecisionTreeClassifier():
    model = compat.convert_sklearn_to_river(
        estimator=tree.DecisionTreeClassifier(
            criterion = 'entropy'
        ),
        classes=[False, True]
    )
    return model

def init_estimator_ExtremelyFastDecisionTreeClassifier():
    model = tree.ExtremelyFastDecisionTreeClassifier(
      grace_period=500, # Number of instances a leaf should observe between split attempts
      delta=0.05, # Significance level (1 - delta) to calculate the Hoeffding bound. Values closer to zero imply longer split decision delays
      nominal_attributes=['feature_0', 'feature_1', 'feature_2'],
      min_samples_reevaluate=500, # Number of instances a node should observe before reevaluating the best split
      max_depth=3
    )
    return model



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    PATH = os.path.dirname(os.path.abspath(__file__))
    data_stream = pd.read_csv(f"{PATH}/data/synthetic_dataset.csv")
    
    #spc_detector = SPCAlgorithm(init_estimator_ExtremelyFastDecisionTreeClassifier) # The estimator should be river compatible! The user is responsible for ensuring this.
    spc_detector = SPCAlgorithm(init_estimator_SGDClassifier) # The estimator should be river compatible! The user is responsible for ensuring this.
    
    metric = metrics.Accuracy()

    report = 1000
    warn = -1
    retrain = -1
    states = [] # for logging
    for i in range(data_stream.shape[0]):
        status, y, y_pred = spc_detector.model_control(data_stream, i)
        metric.update(y, y_pred)

        if (i+1)%report == 0: 
            print(f'{i+1} samples:', metric)

        if status == 'Warning Level' and warn == -1 and i!=0:
            warn = i
            retrain = -1
            states.append(1)
            #print(f'Warning after {i+1} samples')
        elif status == 'Out-control' and retrain == -1 and i!=0:
            #print(f'Re-train model after {i+1} samples')
            retrain = i
            warn = -1
            states.append(2)
        else:
            warn = -1
            retrain = -1
            states.append(0)
    
    print(f'{i+1} samples:', metric)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Plot
    ax1.plot(range(1, len(spc_detector.error_rates) + 1), spc_detector.error_rates, marker='o', label='Error Rate')
    ax1.axhline(y=spc_detector.Pmin + 2 * spc_detector.Smin, color='r', linestyle='--', label='Warning Level')
    ax1.axhline(y=spc_detector.Pmin + 3 * spc_detector.Smin, color='g', linestyle='--', label='Drift Level')
    ax1.set_xlabel('Number of processed samples')
    ax1.set_ylabel('Error rate')
    ax1.set_title('Error Rate Across Processed Samples with SPC Indicators')
    ax1.grid(True)
    ax1.legend()
    # Plotting states
    ax2.plot(range(1, len(states) + 1), states, marker='o', linestyle='-', color='b')
    ax2.set_xlabel('Number of processed samples')
    ax2.set_ylabel('State')
    ax2.set_title('State Across Processed Samples')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Normal', 'Warning', 'Out of Control'])
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

