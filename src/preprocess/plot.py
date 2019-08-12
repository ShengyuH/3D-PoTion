import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def midpoints(x):
    '''Used in the function plot_voxel()'''
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def plot_voxel(data,num=30,size=15,rgb=45):
    '''
    Input data should have the form 3*N, where N is the number of positions

    Parameters
    -------
                num: density of voxels, a larger value will give more voxels so it is smoothier
                size: size of each plotted circle
                rgb: control the color, small value shows more green, large value shows more blue

    Notes
    -------
                Axis scale and labels can be easily customized
    '''


    # define the colors, red -> green -> blue
    colormap = np.array([[3,3,3,3,2,1,0,0,0,0,0,0,0],
                         [0,1,2,3,3,3,3,3,3,3,2,1,0],
                         [0,0,0,0,0,0,0,1,2,3,3,3,3]]) / 3

    # define grid
    center = np.array([size/2,size/2,size/2])
    x,y,z = np.indices((num+1,num+1,num+1)) /num *size
    xc, yc, zc = midpoints(x), midpoints(y), midpoints(z)
    cx, cy, cz = np.zeros((num,num,num)), np.zeros((num,num,num)), np.zeros((num,num,num))

    # attach color to each points according to distance
    stat = np.zeros(13)
    for i in range(num):
        for j in range(num):
            for k in range(num):
                point = np.array([xc[i,j,k],yc[i,j,k],zc[i,j,k]])
                dist = sum((point - center)**2)
                stufe = int(dist // (3*(size/2)**2/rgb))
                if stufe > 12:
                    stufe = 12
                stat[stufe] += 1

                cx[i,j,k], cy[i,j,k], cz[i,j,k] = colormap[0,stufe], colormap[1,stufe], colormap[2,stufe]

    # define the shape
    s1 = (xc - center[0])**2 + (yc - center[1])**2 + (zc - center[2])**2 < (size/2)**2
    s2 = xc+yc+zc < 1.55*size
    s3 = xc+yc+zc > 1.45*size
    sphere = s1*s2*s3

    # assign color
    colors = np.zeros(sphere.shape + (3,))
    colors[..., 0] = cx
    colors[..., 1] = cy
    colors[..., 2] = cz

    # shift data according to sphere center
    data[0] = data[0] - center[0]
    data[1] = data[1] - center[1] 
    data[2] = data[2] - center[2]  

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(data.shape[1]):
        ax.voxels(x+data[0,i], y+data[1,i], z+data[2,i], sphere,
                facecolors=colors,
                #   edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                linewidth=0.5)

    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_xlim3d(0, 100)
    ax.set_ylim3d(0, 100)
    ax.set_zlim3d(0, 100)

    plt.show()


if __name__ == "__main__":

    coors = np.load('./coor.npy')
    plot_voxel(np.array([20,30,40]).reshape(-1,1))
    plot_voxel(coors[:,0,:])
    print('Finished')