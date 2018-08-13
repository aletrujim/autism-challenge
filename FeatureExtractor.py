import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing
#from sklearn.decomposition import PCA

from nilearn.connectome import ConnectivityMeasure


def _load_fmri(fmri_filenames):
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='correlation', vectorize=True))

        
    def fit(self, X_df, y):       
        fmri_filenames = X_df['fmri_power_2011'] 
        self.transformer_fmri.fit(fmri_filenames, y) 
        fmri_filenames2 = X_df['fmri_basc197'] 
        self.transformer_fmri.fit(fmri_filenames2, y)
        fmri_filenames3 = X_df['fmri_craddock_scorr_mean'] 
        self.transformer_fmri.fit(fmri_filenames3, y)
        return self

    def transform(self, X_df):       
        fmri_filenames = X_df['fmri_power_2011']  
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]     
        fmri_filenames2 = X_df['fmri_basc197']
        X_connectome2 = self.transformer_fmri.transform(fmri_filenames2)
        X_connectome2 = pd.DataFrame(X_connectome2, index=X_df.index)
        X_connectome2.columns = ['connectome2_{}'.format(i)
                                for i in range(X_connectome2.columns.size)]   
        fmri_filenames3 = X_df['fmri_craddock_scorr_mean']
        X_connectome3 = self.transformer_fmri.transform(fmri_filenames3)
        X_connectome3 = pd.DataFrame(X_connectome3, index=X_df.index)
        X_connectome3.columns = ['connectome3_{}'.format(i)
                                for i in range(X_connectome3.columns.size)]
                
        X_anatomy = X_df[[col for col in X_df.columns
                          if col.startswith('anatomy')]]       
        
        X_participants = X_df[[col for col in X_df.columns
                          if col.startswith('participants')]]
        
        le = preprocessing.LabelEncoder()
        X_participants['participants_sex'] = le.fit_transform(
                                              X_participants['participants_sex'])
                                      
        concat = pd.concat([X_connectome, X_connectome2, X_connectome3, 
                            X_anatomy, X_participants], axis=1)

        return concat

#test_18
