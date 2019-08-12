from model import *

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, shuffle,path,batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.directory=path
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_flip=int(self.batch_size*0.5)
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        ind_flip=random.sample(list(range(self.batch_size)),self.n_flip)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path=os.path.join(self.directory,ID+'.npy')
            x = np.load(path)
            ind_right=[1,3,5,7,9,11]
            ind_left=[2,4,6,8,10,12]
            if(i in ind_flip and self.shuffle): 
                flipped_x=np.flip(x,1)# flip sample
                for j in range(6):
                    crn_ind_right=ind_right[j]*3
                    crn_ind_left=ind_left[j]*3
                    flipped_x[:,:,[crn_ind_right,crn_ind_right+1,crn_ind_right+2,
                                   crn_ind_left,crn_ind_left+1,crn_ind_left+2]]=flipped_x[:,:,[crn_ind_left,crn_ind_left+1,crn_ind_left+2,crn_ind_right,crn_ind_right+1,crn_ind_right+2]]
                    flipped_x[:,:,[36+ind_right[j],36+ind_left[j]]]=flipped_x[:,:,[36+ind_left[j],35+ind_right[j]]]
                    
                    # crn_ind_right=ind_right[j]*3+64
                    # crn_ind_left=ind_left[j]*3+64
                    # flipped_x[:,:,[crn_ind_right,crn_ind_right+1,crn_ind_right+2,
                    #                crn_ind_left,crn_ind_left+1,crn_ind_left+2]]=flipped_x[:,:,[crn_ind_left,crn_ind_left+1,crn_ind_left+2,crn_ind_right,crn_ind_right+1,crn_ind_right+2]]
                    # flipped_x[:,:,[48+ind_right[j]+64,64+48+ind_left[j]]]=flipped_x[:,:,[64+48+ind_left[j],64+48+ind_right[j]]]

                    # crn_ind_right=ind_right[j]*3+128
                    # crn_ind_left=ind_left[j]*3+128
                    # flipped_x[:,:,[crn_ind_right,crn_ind_right+1,crn_ind_right+2,
                    #                crn_ind_left,crn_ind_left+1,crn_ind_left+2]]=flipped_x[:,:,[crn_ind_left,crn_ind_left+1,crn_ind_left+2,crn_ind_right,crn_ind_right+1,crn_ind_right+2]]
                    # flipped_x[:,:,[128+48+ind_right[j],128+48+ind_left[j]]]=flipped_x[:,:,[128+48+ind_left[j],128+48+ind_right[j]]]
                    
                    
                    
                X[i,]=flipped_x
            else:
                X[i,]=x
            # Store classs
            y[i] = self.labels[ID]-1

        return X, to_categorical(y, num_classes=self.n_classes)