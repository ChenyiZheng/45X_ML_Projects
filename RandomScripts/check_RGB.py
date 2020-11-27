# Test_customDaset error: RuntimeError: The size of tensor a (4) must match the size of tensor b (3) a
# Purpose: Checking if the image is RGB since the CNN input channel = 3
# Manually remove the image from the dataset
# to do: Combine the scripts with generate_csv.py and automatically remove the non RGB images

import os
from PIL import Image

os.chdir(r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset')

for filename in os.listdir():

    im = Image.open(filename)
    if im.mode != 'RGB':
        print(filename)
