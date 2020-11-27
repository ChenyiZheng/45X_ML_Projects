# Test_customDaset error: RuntimeError: The size of tensor a (4) must match the size of tensor b (3) a
# Purpose: Checking if the image is RGB to match the CNN input channel = 3
#          Convert all images to RGB
# to do: Combine the scripts with generate_csv.py

import os
from PIL import Image

os.chdir(r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset')

convert = 0  # conversion flag: print filename and type if flag == 0, convert to RGB if flag == 1

for filename in os.listdir():
    im = Image.open(filename)
    if im.mode != 'RGB':
        print('Image: ' + filename + ' is in ' + im.mode + 'mode')
        if convert == 1:
            im = Image.open(filename).convert('RGB')  # convert image type to RGB
            print('Image ' + filename + ' converted to RGB')
            im.save(filename)

# check if all images are converted to RGB
if convert == 1:
    for filename in os.listdir():
        im = Image.open(filename)
        if im.mode != 'RGB':
            print(filename)
    print('All images in RGB')