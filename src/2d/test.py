from sklearn.metrics import classification_report
from loader import DataGenerator
from configs import *

PATH_DATA='../../potion/pred-32'
MODEL='output/aug-pred-32/best.hdf5'
nChannels=64
RESOLUTION=32

params={
    'dim':(RESOLUTION,RESOLUTION),
    'batch_size':1,
    'n_classes':config['num_classes'],
    'path':PATH_DATA,
    'n_channels':nChannels,
    'shuffle':False
}
input_shape=(RESOLUTION,RESOLUTION,nChannels)

path='../meta/label.json'
with open(path,'r') as f:
    labels=json.load(f)
path='../meta/partition.json'
with open(path,'r') as f:
    partition=json.load(f)
test_generator=DataGenerator(partition['test'],labels,**params)

# get predictions
model=load_model(MODEL)
model.summary()
prediction=model.predict_generator(test_generator)
predict_labels=np.argmax(prediction,axis=1)+1

#  get groundtruth
true_labels=[]
for i in range(len(partition['test'])):
    true_labels.append(labels[partition['test'][i]])
true_labels=np.array(true_labels)[:predict_labels.shape[0]]
print(classification_report(true_labels,predict_labels,digits=4))


