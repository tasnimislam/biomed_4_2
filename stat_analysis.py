from sklearn.decomposition import PCA
from image_processing import *

'''
TODO: sklearn.decomposition.TruncatedSVD
'''
def get_pca(one_d_array, n_components):
    '''
    one_d_array: one dimentional array of which the PCA should be done
    The estimated number of components. When n_components is set to ‘mle’ or a number between 0 and 1 (with svd_solver == ‘full’)
    this number is estimated from input data.
    Otherwise it equals the parameter n_components, or the lesser value of n_features and n_samples if n_components is None.
    '''
    pca = PCA(n_components = n_components)
    new_array = pca.fit_transform(one_d_array) #get the new transformed PCA array
    return new_array