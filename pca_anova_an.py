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
        class_0.append(pca_data[i, :])
    if l==1:
        class_1.append(pca_data[i, :])
    # if l==2:
    #     class_2.append(pca_data[i, :])
    # if l==3:
    #     class_3.append(pca_data[i, :])
    # if l==4:
    #     class_4.append(pca_data[i, :])



stat, p_value = f_oneway(np.array(class_0), np.array(class_1)) #,np.array(class_2), np.array(class_3), np.array(class_4))
p_value_without_nan = p_value[np.logical_not(np.isnan(p_value))]
print(p_value_without_nan)

# """
# TerminalFeature
# [0.57918159 0.83380661 0.14695914 0.58720784 0.24988438 0.15213255 0.40873347 0.42471912 0.88326108] - many classes
# [0.49392536 0.4751084  0.05321809 0.37790133 0.53715939 0.05765857 0.71218432 0.69545349 0.76487285] - 0 vs other classes






