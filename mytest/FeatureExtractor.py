import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from nilearn.connectome import ConnectivityMeasure


def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values for subject_filename in fmri_filenames])

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='tangent', vectorize=False))

    def fit(self, X_df, y):
        fmri_filenames = X_df['fmri_power_2011']
        self.transformer_fmri.fit(fmri_filenames, y)

        return self

    def transform(self, X_df):
        fmri_filenames = X_df['fmri_power_2011']
        X_connectomes = self.transformer_fmri.transform(fmri_filenames)

        flattened_connectomes = []
        indices_to_keep = np.triu_indices(X_connectomes.shape[1])
        for i in range(0, X_connectomes.shape[0]):
            tmp = X_connectomes[i, :, :]
            flattened_connectomes.append(np.ndarray.flatten(tmp[indices_to_keep]))

        X_connectomes = pd.DataFrame(flattened_connectomes, index=X_df.index)
        X_connectomes.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectomes.columns.size)]

        return pd.concat([X_connectomes, X_df["fmri_select"]], axis=1)
