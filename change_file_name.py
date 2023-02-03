import os
from glob import glob

def change_file_names(folder_name, file_name_old, file_name_new):
    file_names = glob(f'{folder_name}/*/{file_name_old}')
    for fil in file_names:
        person_name = fil.split('/')[-1].split('\\')[-2]
        print(f'{folder_name}/{person_name}/{file_name_new}')
        os.rename(fil, f'{folder_name}/{person_name}/{file_name_new}')

