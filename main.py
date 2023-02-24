import numpy as np

from image_processing import *
from data_and_label_processing import *
from model import *
import matplotlib.pyplot as plt
from change_file_name import  *
from stat_analysis import *

# folder_name = "D:/4-2/biomedical project/Fingerprint"
# file_name_old = 'tt.bmp'
# file_name_new = 'lt.bmp'
# change_file_names(folder_name, file_name_old, file_name_new)


# Image processing
file_name = 'D:/4-2/biomedical project/Fingerprint/1711046(b+)/ri.bmp'
image = image_preprocessing(file_name)
# plt.imshow(image, cmap = 'binary')
# plt.show()
#
# # # pandas data processing
# # data_path = 'D:/4-2/biomedical project/sheet_name_blood_group.xlsx'
# # excel_file = read_xlx_file(data_path)
# # print(excel_file)
#
#
# prepare the dataset
data_path = 'D:/4-2/biomedical project/sheet_name_blood_group(all).xlsx'
universal_path = 'D:/4-2/biomedical project/Fingerprint'
class_used = [0, 1, 2, 3, 4]
data, label = get_data_label_alltogether(data_path, universal_path, class_used)
print(data.shape, label.shape)
# #
# # # train test split
# # (X_train, y_train), (X_test, y_test) = train_test_split_custom(data, label, 4)
# # assert len(X_train) == len(y_train)
# # print(len(X_train), len(X_test))
# # print(y_train.shape)
#
# train
# data_path = 'D:/4-2/biomedical project/sheet_name_blood_group(all).xlsx'
# universal_path = 'D:/4-2/biomedical project/Fingerprint'
# class_no = 5
# epoch_no = 50
# class_used = [2, 0]
# for i in range(10):
#     run_number = i
#     model = train_sklearn(data_path, universal_path,class_no, epoch_no, class_used, run_number)


# Image feature
skel, mask = get_skel_mask(image)
(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);
FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif)
BifLabel = skimage.measure.label(minutiaeBif, connectivity=1);
TermLabel = skimage.measure.label(minutiaeTerm, connectivity=1);

transformed_biflab = get_pca(np.reshape(data, (91, -1)), 3)
print(transformed_biflab.shape)

plt.figure()
plt.subplot(211)
plt.plot(transformed_biflab[10, :])
plt.subplot(212)
plt.imshow(TermLabel)
plt.show()

