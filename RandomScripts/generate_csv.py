import csv
import os

csv_path = r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset'
image_path = r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset'
os.chdir(image_path)

flag = 'non'

with open(os.path.join(csv_path, 'fire_dataset_complete.csv'), 'w', newline='') as f:
    thewriter = csv.writer(f)
    for filename in os.listdir():
        if flag in filename:
            index = '1'
        else:
            index = '0'
        thewriter.writerow([filename, index])
