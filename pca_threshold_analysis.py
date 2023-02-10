import numpy as np

from image_processing import *
from data_and_label_processing import *
from model import *
import matplotlib.pyplot as plt
from change_file_name import  *
from stat_analysis import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def average_array(arr, n):
    avg_arr = []
    for i in range(0, len(arr), n):
        avg = sum(arr[i:i + n]) / n
        avg_arr.append(avg)
    return avg_arr

# Image processing
file_name = 'D:/4-2/biomedical project/Fingerprint/1711046(b+)/ri.bmp'
image = image_preprocessing(file_name)

data_path = 'D:/4-2/biomedical project/sheet_name_blood_group(all).xlsx'
universal_path = 'D:/4-2/biomedical project/Fingerprint'
class_used = [0, 1, 2, 3, 4]
data, label = get_data_label_alltogether(data_path, universal_path, class_used)
print(data.shape, label.shape)

data = np.reshape(data, (91, -1))
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(data)

pca = PCA().fit(data_rescaled)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 11, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
y = average_array(y, 10)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()
