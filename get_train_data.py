from PIL import Image
import os
import numpy as np

test_data_folder = 'test_data'
train_data_foler = 'train_data'

train_data_files = os.listdir(train_data_foler)
train_data = []

for train_data_file in train_data_files:
  train_image = Image.open(os.path.join(train_data_foler, train_data_file))
  train_data.append({
      "label": train_data_file[0:4],
      "image": train_image.crop((20, 1, 117, 25))
  })
  train_data.append({
      "label": train_data_file[4:8],
      "image": train_image.crop((20, 25, 117, 49))
  })

# first_image = Image.open(os.path.join(train_data_foler, train_data_files[77]))
# top_half_image = first_image.crop((20, 1, 117, 25))
# top_half_image.show()
# print(top_half_image.size)

# bottom_half_image = first_image.crop((20, 25, 117, 49))
# bottom_half_image.show()
# print(bottom_half_image.size)

train_data[56]["image"].show()
print(train_data[56]["label"])
