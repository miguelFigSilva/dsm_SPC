from SPC_module.SPC import SPCAlgorithm
import pandas as pd
import os
import warnings
from river import compat
from river import metrics
from river import preprocessing
from sklearn import linear_model
from river import tree
  


def init_estimator_SGDClassifier():
    model = preprocessing.StandardScaler()
    model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(
        #loss='log_loss',                   # 'log_loss' gives LOGISTIC REGRESSION. Uncomment to use it.
        #loss='hinge',                     # 'hinge' gives a LINEAR SVM. Uncomment to use it.
        loss='perceptron',                # 'perceptron' is the linear loss used by the PERCEPTRON algorithm. Uncomment to use it.
        #loss='modified_huber',            # 'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates. Uncomment to use it.
        #loss='squared_hinge',             # 'squared_hinge' is like hinge but is quadratically penalized. Uncomment to use it.
        eta0=0.01,
        learning_rate='constant'
    ),
    classes=[False, True]
    )
    return model

def init_estimator_DecisionTreeClassifier():    # not working : tree has no DecisionTreeClassifier
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
    
    # The estimator should be river compatible! The user is responsible for ensuring this.
    spc_detector = SPCAlgorithm(init_estimator_SGDClassifier) 
    #spc_detector = SPCAlgorithm(init_estimator_ExtremelyFastDecisionTreeClassifier)
    
    metric = metrics.Accuracy()

    report = 1000
    warn = -1
    retrain = -1
    for i in range(data_stream.shape[0]):
        status, y, y_pred = spc_detector.model_control(data_stream, i)        
        metric.update(y, y_pred)

        if (i+1)%report == 0: 
            print(f'{i+1} samples:', metric)

        if status == 'Warning Level' and warn == -1 and i!=0:
            warn = i
            retrain = -1
            #print(f'Warning after {i+1} samples')
        elif status == 'Out-control' and retrain == -1 and i!=0:
            #print(f'Re-train model after {i+1} samples')
            retrain = i
            warn = -1
        else:
            warn = -1
            retrain = -1
    
    print(f'{i+1} samples:', metric)

    # Plotting
    spc_detector.process_plot()

