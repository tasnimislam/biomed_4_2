import numpy as np

from image_processing import *
from data_and_label_processing import *
from model import *
import matplotlib.pyplot as plt
from change_file_name import  *
from stat_analysis import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway
import scipy.stats as stats

# Image processing
file_name = 'D:/4-2/biomedical project/Fingerprint/1711046(b+)/ri.bmp'
image = image_preprocessing(file_name)

data_path = 'D:/4-2/biomedical project/sheet_name_blood_group(all).xlsx'
universal_path = 'D:/4-2/biomedical project/Fingerprint'
class_used = [0, 1, 2, 3, 4]
data, label = get_data_label_alltogether(data_path, universal_path, class_used)

data = np.reshape(data, (116, -1))
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(data)

pca_data = PCA(n_components=9).fit_transform(data_rescaled)
print(pca_data.shape)

class_0 = []
class_1 = []
class_2 = []
class_3 = []
class_4 = []

for i, l in enumerate(label):
    if l==0:
        class_0.append(data[i, :])
    if l==1:
        class_1.append(data[i, :])
    if l==2:
        class_2.append(data[i, :])
    if l==3:
        class_3.append(data[i, :])
    if l==4:
        class_4.append(data[i, :])

stat, p_value = f_oneway(np.array(class_0), np.array(class_1), np.array(class_2), np.array(class_3), np.array(class_4))
p_value_without_nan = p_value[np.logical_not(np.isnan(p_value))]
print(stats.describe(p_value_without_nan))

p_val_insignificant = p_value_without_nan[p_value_without_nan<0.05000000000000000000000000000000001]
print(stats.describe(p_val_insignificant))
print(p_val_insignificant.shape)

# """
# Result: F_onewayResult(statistic=array([nan, nan, nan, ..., nan, nan, nan]), pvalue=array([nan, nan, nan, ..., nan, nan, nan]))
# F_onewayConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite warnings.warn(F_onewayConstantInputWarning())
# ConstantInputWarning
#     Raised if all values within each of the input arrays are identical. In this case
#     the F statistic is either infinite or isn’t defined, so np.inf or np.nan is returned.
#
# DegenerateDataWarning
#     Raised if the length of any input array is 0, or if all the input arrays have
#     length 1. np.nan is returned for the F statistic and the p-value in these cases.
#
# TermLabel
#
#
# When trained in raw data
# C:\Users\manami\anaconda3\lib\site-packages\scipy\stats\_stats_py.py:3659: F_onewayConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
#   warnings.warn(F_onewayConstantInputWarning())
# (73728,)
# (73728,)

# p_value in raw data the array summary = DescribeResult(nobs=73535, minmax=(8.192564419216128e-08, 0.9999878165177017), mean=0.4965325837742704, variance=0.08214839083494557, skewness=0.017657465153195336, kurtosis=-1.195087960718649)
# p_value <0.05 summary = DescribeResult(nobs=3579, minmax=(8.192564419216128e-08, 0.04998453519672729), mean=0.02431651275866173, variance=0.00021481994670352326, skewness=0.030492857563073794, kurtosis=-1.224955033931781)
# p_value <0.05 shape 3579
# n components = 9





