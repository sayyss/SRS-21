import os

os.system('git clone https://github.com/fchollet/ARC; mv ./ARC/data ./; rm -rf ARC')

from arc_vae.preprocess_data.tagger import tag
from arc_vae.preprocess_data.data_augmentation import augment_data

tag()
augment_data()



