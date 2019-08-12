from utils_2d import *
from config import configs

# get 2D potion from ground truth, The input paths are PennAction directory and output directory
PATH_PENN=configs['dir_penn_action']
PATH_OUTPUT=os.path.join(configs['dir_potion'],'gt-'+str(configs['frame_size']))
if(os.path.exists(PATH_OUTPUT)):
    os.system('rm -r '+PATH_OUTPUT)
os.system('mkdir '+PATH_OUTPUT)
DIR_LABELS=os.path.join(PATH_PENN,'labels')

NUM_JOINT=configs['num_joints']['gt']
FRAME_SIZE=configs['frame_size']
C=configs['C']

# multiprecessing worker to deal with each video
def worker(each_label):
    try:
        print(each_label)
        path=os.path.join(DIR_LABELS,each_label)
        annotation=sio.loadmat(path)
        label=dict_action[annotation['action'][0]]
        isTrain=annotation['train'][0][0]
        nFrames=annotation['nframes'][0][0]
        joints_x=annotation['x']
        joints_y=annotation['y']
        min_x=np.min(joints_x)-5
        min_y=np.min(joints_y)-5
        max_x=np.max(joints_x)+5
        max_y=np.max(joints_y)+5
        scale_x=FRAME_SIZE/(max_x-min_x)
        scale_y=FRAME_SIZE/(max_y-min_y)
        scale=min(scale_x,scale_y)
        
        # normalize all the joints to have desired FRAME_SIZE and FRAME_SIZE
        normalized_x=(joints_x-min_x)*scale
        normalized_y=(joints_y-min_y)*scale
        
        normalized_x=normalized_x.astype('int')
        normalized_y=normalized_y.astype('int')
        
        # generate ramdom values for this video
        rand_amplitudes=randint(85,100,nFrames*NUM_JOINT)/100
        rand_stddev_x=randint(15,25,nFrames*NUM_JOINT)/10
        rand_stddev_y=randint(15,25,nFrames*NUM_JOINT)/10
        
        # get heatmap
        heatmaps=np.zeros((nFrames,NUM_JOINT,FRAME_SIZE,FRAME_SIZE))
        ind=0
        for i in range(nFrames):
            for j in range(NUM_JOINT):
                g2d = Gaussian2D(amplitude=rand_amplitudes[ind],x_mean=normalized_y[i,j], y_mean=normalized_x[i,j], 
                        x_stddev=rand_stddev_x[ind], y_stddev=rand_stddev_y[ind]) 
                ind+=1
                heatmaps[i,j,:,:]=g2d(*np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE])
                
        # get descriptor
        each_label=each_label.split('.')[0]
        descriptor=get_descriptor(heatmaps,C).astype('float32')

        # keras take channel last
        descriptor=np.moveaxis(descriptor,0,-1)
        path=os.path.join(PATH_OUTPUT,each_label+'.npy')
        np.save(path,descriptor)
    except:
        print('Error'+ path)

if __name__=='__main__':
    # Get potion representation
    all_labels=sorted(os.listdir(DIR_LABELS)) # list all the videos
    p=Pool()
    for each_label in all_labels:  # loop each video
        if(each_label!='.DS_Store'): # mac may corrupt the file
            p.apply_async(worker,args=(each_label,))
    p.close()
    p.join()  # wait for finishing all the jobs

    # get meta data
    partition={}
    partition['train']=[]
    partition['test']=[]
    labels={}

    for each_label in all_labels:
        if(each_label!='.DS_Store'): 
            path=os.path.join(DIR_LABELS,each_label)
            annotation=sio.loadmat(path)
            label=dict_action[annotation['action'][0]]
            each_label=each_label.split('.')[0]
            isTrain=annotation['train'][0][0]
            # save the dictionary
            labels[each_label]=label
            if(isTrain==1):
                partition['train'].append(each_label)
            else:
                partition['test'].append(each_label)

    # save meta data
    os.system('mkdir '+PATH_OUTPUT+'/meta')
    path=os.path.join(PATH_OUTPUT,'meta','partition.json')
    with open(path,'w') as fp:
        json.dump(partition,fp)

    path=os.path.join(PATH_OUTPUT,'meta','labels.json')
    with open(path,'w') as fp:
        json.dump(labels,fp)