from configs import *
# remember that the input data should always be channels last

def build_model(config,input_shape):
    config['input_shape']=input_shape
    model=Sequential()
    model.add(Conv2D(filters = config['n_filters'][0],
                        padding="same",
                        kernel_size = (config['kernel_size'],config['kernel_size']),
                        kernel_initializer=config['kernel_initializer'],
                        strides=(config['strides'][0],config['strides'][0]),
                        input_shape =config['input_shape']))
    model.add(BatchNormalization())
    model.add(Activation(config['activation']))

    for i in range(1,len(config['n_filters'])):
        model.add(Conv2D(filters = config['n_filters'][i] , 
                        padding="same",
                        kernel_size = (config['kernel_size'],config['kernel_size']),
                        kernel_initializer=config['kernel_initializer'],
                        strides=(config['strides'][i],config['strides'][i])))
        model.add(BatchNormalization())
        model.add(Activation(config['activation']))
    model.add(GlobalAveragePooling2D(data_format=config['data_format']))
    return model