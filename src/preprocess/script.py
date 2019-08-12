import json
import os,sys
from collections import Counter
import random
import numpy as np
dir_meta='../../potion/gt-32/meta'
path=os.path.join(dir_meta,'partition.json')
with open(path,'r') as f:
    partition=json.load(f)
path=os.path.join(dir_meta,'labels.json')
with open(path,'r') as f:
    label=json.load(f)
train_labels=[label[ele] for ele in partition['train']]
num_dict=Counter(train_labels)

num_training_samples=len(partition['train'])
num_validation=int(num_training_samples*0.3)
ind_validation=random.sample(list(range(num_training_samples)),num_validation)
ind_train=list(set(list(range(num_training_samples)))-set(ind_validation))

new_partition={}
new_partition['train']=[partition['train'][ele] for ele in ind_train]
new_partition['test']=partition['test']
new_partition['validation']=[partition['train'][ele] for ele in ind_validation]
print(len(new_partition['train']))
print(len(new_partition['test']))
print(len(new_partition['validation']))

validation_label=[label[ele] for ele in new_partition['validation']]

print(len(new_partition['train']))
print(len(new_partition['test']))
print(len(new_partition['validation']))

# path=os.path.join('../meta','partition.json')
# with open(path,'w') as fp:
#     json.dump(new_partition,fp)
    
# path=os.path.join('../meta','label.json')
# with open(path,'w') as fp:
#     json.dump(label,fp)