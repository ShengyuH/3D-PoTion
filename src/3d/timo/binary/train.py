from model import *
from loader import DataGenerator

init_lr=1e-2
def scheduler(epoch):
    if epoch % 20==0 and epoch!=0:
        K.set_value(model.optimizer.lr,init_lr/(1+0.025*epoch))
    return K.get_value(model.optimizer.lr)

PATH_DATA = config['dir_potion']
MODEL = config['model_name']
nChannels = config['n_channels']
FRAME_SIZE = config['frame_size']

params = {
    'dim': (FRAME_SIZE, FRAME_SIZE, FRAME_SIZE),
    'batch_size': config['batch_size'],
    'n_classes': config['num_classes'],
    'path': PATH_DATA,
    'n_channels': nChannels,
}
input_shape = (FRAME_SIZE, FRAME_SIZE, FRAME_SIZE, nChannels)

path = os.path.join('../meta/label.json')
with open(path, 'r') as f:
    labels = json.load(f)
path = os.path.join('../meta/partition.json')
with open(path, 'r') as f:
    partition = json.load(f)

training_generator=DataGenerator(partition['train'],labels,True,**params)
test_generator=DataGenerator(partition['test'],labels,False, **params)
validation_generator=DataGenerator(partition['validation'],labels,False, **params)

model=timo(config,input_shape)

# optimizer=Adam(lr=config['lr'])   # optimizer
optimizer=SGD(lr=0.003,momentum=0.9,nesterov=True)

model.compile(optimizer=optimizer,loss=config['loss'],metrics=["accuracy"])    # compile the model

reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, mode='auto',verbose=1)
# reduce_lr=LearningRateScheduler(scheduler) # learning rate schedule

path_output=os.path.join(config['output'],'binary')
if(os.path.exists(path_output)):
    os.system('rm -r '+path_output)
os.system('mkdir '+path_output)
os.system('cp configs.py '+path_output)
os.system('cp model.py '+path_output)
os.system('cp train.py '+path_output)

model_path=os.path.join(path_output,'best.hdf5')
checkpoint=ModelCheckpoint(monitor='val_loss',filepath=model_path,save_best_only=True,verbose=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
history=model.fit_generator(generator=training_generator,
    epochs=config['epoch'],
    validation_data=validation_generator,
    use_multiprocessing=False,
    callbacks=[TensorBoard(log_dir=path_output),checkpoint,reduce_lr,earlystopping])

prediction=model.predict_generator(test_generator)
predict_labels=np.argmax(prediction,axis=1)+1

#  get groundtruth
true_labels=[]
for i in range(len(partition['test'])):
    true_labels.append(labels[partition['test'][i]])
true_labels=np.array(true_labels)[:predict_labels.shape[0]]
print(classification_report(true_labels,predict_labels,digits=4))