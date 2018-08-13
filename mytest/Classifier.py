import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class Classifier(BaseEstimator):
    def __init__(self):         
        self.clf_connectome = make_pipeline(StandardScaler(),
                                            MLPClassifier(alpha=1))
        self.clf_connectome2 = make_pipeline(StandardScaler(),
                                            MLPClassifier(alpha=1))  
        self.clf_connectome3 = make_pipeline(StandardScaler(),
                                            MLPClassifier(alpha=1))
        self.clf_anatomy = make_pipeline(StandardScaler(),
                                         MLPClassifier(alpha=1))       
        self.clf_participants = make_pipeline(StandardScaler(),
                                         MLPClassifier(alpha=1))
        self.meta_clf = MLPClassifier(alpha=1)
        

    def fit(self, X, y):        
        X_anatomy = X[[col for col in X.columns 
                       if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]
        X_connectome2 = X[[col for col in X.columns
                          if col.startswith('connectome2')]]
        X_connectome3 = X[[col for col in X.columns
                          if col.startswith('connectome3')]]
        X_participants = X[[col for col in X.columns
                            if col.startswith('participants')]]
        
        train_idx, validation_idx = train_test_split(range(y.size),
                                                     test_size=0.33, 
                                                     shuffle=True,
                                                     random_state=None) 
        
        X_anatomy_train = X_anatomy.iloc[train_idx]
        X_anatomy_validation = X_anatomy.iloc[validation_idx]
        X_connectome_train = X_connectome.iloc[train_idx]
        X_connectome_validation = X_connectome.iloc[validation_idx]
        X_connectome2_train = X_connectome2.iloc[train_idx]
        X_connectome2_validation = X_connectome2.iloc[validation_idx]
        X_connectome3_train = X_connectome3.iloc[train_idx]
        X_connectome3_validation = X_connectome3.iloc[validation_idx]
        X_participants_train = X_participants.iloc[train_idx]
        X_participants_validation = X_participants.iloc[validation_idx]
        
        y_train = y[train_idx]
        y_validation = y[validation_idx]

        self.clf_connectome.fit(X_connectome_train, y_train)
        self.clf_connectome2.fit(X_connectome2_train, y_train)
        self.clf_connectome3.fit(X_connectome3_train, y_train)
        self.clf_anatomy.fit(X_anatomy_train, y_train)
        self.clf_participants.fit(X_participants_train, y_train)

        y_connectome_pred = self.clf_connectome.predict_proba(
            X_connectome_validation)
        y_connectome2_pred = self.clf_connectome2.predict_proba(
            X_connectome2_validation)
        y_connectome3_pred = self.clf_connectome3.predict_proba(
            X_connectome3_validation)
        y_anatomy_pred = self.clf_anatomy.predict_proba(
            X_anatomy_validation)
        y_participants_pred = self.clf_participants.predict_proba(
            X_participants_validation)

        self.meta_clf.fit(
            np.concatenate([y_connectome_pred, y_connectome2_pred, y_connectome3_pred, 
                            y_anatomy_pred, y_participants_pred], axis=1),
            y_validation)
        
        return self
    
    def predict(self, X):
        X_anatomy = X[[col for col in X.columns 
                       if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]
        X_connectome2 = X[[col for col in X.columns
                          if col.startswith('connectome2')]]
        X_connectome3 = X[[col for col in X.columns
                          if col.startswith('connectome3')]]
        X_participants = X[[col for col in X.columns
                            if col.startswith('participants')]]        

        y_anatomy_pred = self.clf_anatomy.predict_proba(X_anatomy)
        y_connectome_pred = self.clf_connectome.predict_proba(X_connectome)
        y_connectome2_pred = self.clf_connectome2.predict_proba(X_connectome2)
        y_connectome3_pred = self.clf_connectome3.predict_proba(X_connectome3)
        y_participants_pred = self.clf_participants.predict_proba(X_participants)

        return self.meta_clf.predict(
            np.concatenate([y_connectome_pred, y_connectome2_pred, y_connectome3_pred, 
                            y_anatomy_pred, y_participants_pred], axis=1))

    def predict_proba(self, X):
        X_anatomy = X[[col for col in X.columns 
                       if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]
        X_connectome2 = X[[col for col in X.columns
                          if col.startswith('connectome2')]]
        X_connectome3 = X[[col for col in X.columns
                          if col.startswith('connectome3')]]
        X_participants = X[[col for col in X.columns
                            if col.startswith('participants')]]

        y_anatomy_pred = self.clf_anatomy.predict_proba(X_anatomy)
        y_connectome_pred = self.clf_connectome.predict_proba(X_connectome)
        y_connectome2_pred = self.clf_connectome2.predict_proba(X_connectome2)
        y_connectome3_pred = self.clf_connectome3.predict_proba(X_connectome3)
        y_participants_pred = self.clf_participants.predict_proba(X_participants)

        return self.meta_clf.predict_proba(
            np.concatenate([y_connectome_pred,y_connectome2_pred, y_connectome3_pred, 
                            y_anatomy_pred, y_participants_pred], axis=1))
      
#test_18
