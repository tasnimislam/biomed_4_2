import numpy as np

from image_processing import *
from data_and_label_processing import *
from model import *
import matplotlib.pyplot as plt
from change_file_name import  *
from stat_analysis import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Image processing
file_name = 'D:/4-2/biomedical project/Fingerprint/1711046(b+)/ri.bmp'
image = image_preprocessing(file_name)

data_path = 'D:/4-2/biomedical project/sheet_name_blood_group(all).xlsx'
universal_path = 'D:/4-2/biomedical project/Fingerprint'
class_used = [0, 1, 2, 3, 4]
data, label = get_data_label_alltogether(data_path, universal_path, class_used)
print(data.shape, label.shape)


data = np.reshape(data, (116, -1))
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(data)

pca_data = PCA(n_components=0.95).fit_transform(data_rescaled)
print(pca_data.shape)