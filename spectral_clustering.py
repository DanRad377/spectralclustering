import networkx as nx
import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from PIL import Image

# This allows the use of larger images.
mpl.rcParams['agg.path.chunksize'] = 1000000000000

def data_storage(description,parameters):
    '''
    Description
    ----------
    This function checks if there is a folder for the data you are working with.
    If there is not, it makes a folder.
    
    The folder is .\Data\description\...
    
    Parameters
    ----------
    description    : string
                     Give a description of the data you will be collecting.
    parameters     : string
                     Give any parameters to describe the data further.

    Returns
    -------
    directory : string
        Returns the folder location.

    '''
    
    directory = 'Data'
    
    if directory not in os.listdir():
        os.mkdir(directory)
        print("Created directory '{}'.".format(directory))
    
    if description not in os.listdir(directory):
        os.mkdir(directory + '//{}'.format(description))
        print("Created directory '{}' in '{}'.".format(description, directory))
    directory = directory + '//{}'.format(description)
    
    if parameters not in os.listdir(directory):
        os.mkdir(directory + '//{}'.format(parameters))
        print("Created directory '{}' in '{}'.".format(parameters,directory))
    directory = directory + '//{}'.format(parameters)
    print('Saving data in {}.'.format(directory))
    
    return directory

def img_to_mtx(file):
    '''
    Description
    ----------
    Converts an image file into a matrix in the form of a numpy array.
    
    Parameters
    ----------
    file        : string
                  Image file name including path relative to current working
                  directory.

    Returns
    -------
    img_matrix  : numpy array
                  An n*m*k array of the image. n and m are the height and width
                  dimensions of the image. k is equal to the number of components in
                  the color palette of the image.

    '''
    img = Image.open(file)
    img_matrix = np.asarray(img).astype('float64')
    
    return img_matrix

def mtx_to_img(arr,name="Saved_Image"):
    '''
    Description
    ----------
    Converts a matrix to a png image file. This function creates a directory
    named "Output Images" in the current working directory if one does not
    already exist. The image is saved there.
    
    If there is already a file with that name, this will add an integer at the
    end of the filename until it is unique.
    
    Parameters
    ----------
    arr     : numpy array
              Given an array with shape n*m*k, the saved image will be n pixels
              high, m pixels wide, and the color palette will have k components.
    name    : string
              This is the name you wish to give to the file.
        
    '''
    
    if 'Output Images' not in os.listdir():
        os.mkdir('Output Images')
    
    im = Image.fromarray(arr.astype('uint8'))
    
    i = 0
    file = name
    
    while "{}.png".format(file) in os.listdir("Output Images"):
        i += 1
        file = "{}{}".format(name,i)
    
    im.save("Output Images//{}.png".format(file))

def color_sim(arr,alpha=1): # color similarity
    '''
    Description
    ----------
    This function compares each pair of datapoints by the component values.
    Distance is found by using the Euclidean metric.
    
    A pair of points with a low distance are considered highly similar.
    
    This function returns an array w which, for any i and j, the entry in i,j is
    the distance between datapoints i and j.
    
    This function also returns r and c which are the dimensions of the raw data.
    These are used in other functions.
    
    Parameters
    ----------
    arr     : numpy array
              raw data array
    alpha   : float, optional
              This is for scaling the data. The default is 1. The norm of the data
              will be this value before being used as exponent. Must be positive.

    Returns
    -------
    w       : numpy array
              Color weights of the array. The array is (r*c)x(r*c)
    r       : int
              Number of rows in arr.
    c       : int
              Number of columns in arr.

    '''
    
    try:
        r,c,dim = arr.shape[0],arr.shape[1],arr.shape[2]
    except IndexError:
        r,c,dim = arr.shape[0],arr.shape[1],1
    
    row = arr.reshape(r*c,1,dim)
    col = arr.reshape(1,r*c,dim)
    
    diff = row-col
    
    power = np.linalg.norm(diff,axis = -1)
    
    if alpha <= 0:
        raise ValueError("Value alpha must be positive.")
    
    const = alpha
    
    w = np.e**(-(power**2)/const)
    
    return w, r, c

def grid_dist(row,col):
    '''
    Description
    ----------
    This function returns a square numpy array with side length row*col. The
    entries are the distances between positions in a row*col grid.
    
    Parameters
    ----------
    row     : integer
              Number of rows in grid.
    col     : integer
              Number of columns in grid.

    Returns
    -------
    mag_grid : numpy array
               An array of size row*col.
               Entries are the distances between two positions of the grid.

    '''
    
    row_pos = np.array(range(row))
    col_pos = np.array(range(col))
    
    grid_pos = np.array(np.meshgrid(row_pos,col_pos)).T.reshape(row*col,1,2)
    grid_posT = grid_pos.reshape(1,row*col,2)
    
    diff = grid_pos-grid_posT
    
    mag_grid = np.linalg.norm(diff,axis=-1)
    
    return mag_grid

def loc_sim(row,col,dist=1,alpha=1): # location similarity
    '''
    Description
    ----------
    This function compares each pair of datapoints by their positions in the
    dataset. Distance is found by using the Euclidean metric.
    
    A pair of points with a low distance are considered highly similar.
    
    This function returns an array dist_mtx which, for any i and j, the entry
    in i,j is the distance between datapoints i and j.
    
    Distances greater than dist are considered unrelated.
    
    Parameters
    ----------
    row     : integer
              Number of rows in grid.
    col     : integer
              Number of columns in grid.
    dist    : float, optional
              Maximum distance between grid positions. The default is 1.
    alpha   : float, optional
              This is for scaling the data. The default is 1. The norm of the
              data will be this value before being used as exponent. Must be
              positive.

    Returns
    -------
    dist_mtx : numpy array
               Location weights of the grid. The array is (r*c)x(r*c)

    '''
    
    power = grid_dist(row,col)
    
    if alpha <= 0:
        raise ValueError("Value alpha must be positive.")
    
    dist_mtx = np.e**(-(power**2)/alpha)
    dist_mtx = dist_mtx * (power != 0) * (power<=dist)
    
    return dist_mtx

def img_to_csv(infile,dist=1,c_alpha=1,l_alpha=1):
    '''
    Description
    ----------
    This function combines previous functions to create a weight matrix for the
    image. The weights are created by finding the Euclidean distances between
    the datapoints and the Euclidean distances between their positions in the
    dataset.
    
    The infile is an image. Pixels are considered related if the distance
    between them is at most dist.
    
    This function saves the weight matrix in ./Data. It returns the path of the
    matrix. It also returns the raw data and its dimensions.
        
    Parameters
    ----------
    infile      : string
                  Image filename.
    dist        : float, optional
                  Maximum distance between grid positions. The default is 1.
    c_alpha     : float, optional
                  This is for scaling the data. The default is 1. The norm of
                  the color component of the data will be this value before
                  being used as exponent. Must be positive.
    l_alpha     : float, optional
                  This is for scaling the data. The default is 1. The norm of
                  the data will be this value before being used as exponent.
                  Must be positive.

    Returns
    -------
    outfileLOC  : string
                  File location for data.
    r           : integer
                  Number of rows in grid.
    c           : integer
                  Number of columns in grid.
    img_mtx     : numpy array
                  An n*m*k array of the image. k is dependent on the color
                  palette of the image. 

    '''
    
    img_mtx = img_to_mtx(infile)
    outfile = os.path.splitext(os.path.split(infile)[-1])[0]
    parameters = '(D={},Ca={},La={})'.format(dist,c_alpha,l_alpha)
    
    outfileLOC = data_storage(outfile,parameters)
    
    w, r, c = color_sim(img_mtx,
                        alpha=c_alpha)
    
    adj = loc_sim(r,
                  c,
                  dist=dist,
                  alpha=l_alpha)
    w = w*adj
    
    w_df = pd.DataFrame(w)
    
    outfileNAME = 'weights_{}{}.csv'.format(outfile,parameters)
    outfilePATH = '{}//{}'.format(outfileLOC,outfileNAME)
    
    w_df.to_csv(outfilePATH)
    print('Weights saved as {} in {} directory.'.format(outfileNAME, outfileLOC))
    
    return outfilePATH, r, c, img_mtx

def csv_to_G(file,thhld=0,draw=False):
    '''
    Description
    ----------
    Given a weight matrix from a csv file, this function will create a networkx
    graph of the data.
    
    The thhld parameter is a threshold for the minimum value of weights. Any edge
    with weight under thhld% of the maximum weight is removed from the graph.
    By default, this parameter is set to 0, so it does not remove any edges.
    
    The function returns the graph and draws it if draw is True.
    
    Parameters
    ----------
    file    : string
              A CSV file containing the weights of edges connecting from a
              dataset.
    thhld   : float, optional
              Should be a value in [0,100]. This is used to exclude lighter
              weights
              from the graph. The default is 0.
    draw    : boolean, optional
              True draws the graph. The default is False.

    Raises
    ------
    ValueError
        thhld must be a float between 0 and 100.

    Returns
    -------
    G       : networkx graph
              Networkx graph of data.

    '''
    
    # Collect Weights from CSV
    weights = pd.read_csv(file,
                          index_col=0).values
    
    # Removing the Light Weights
    try:
        thhld = float(thhld)
    except ValueError and TypeError:
        raise ValueError("Threshold value must be a float.")
    
    if (thhld < 0) or (thhld > 100):
        raise ValueError("Please use numeric values from 0 to 100.")
        
    weights = weights * (weights/np.max(weights) >= thhld/100)
    
    # create the graph
    G = nx.Graph(weights)
    if draw:
        nx.draw(G)
    
    return G

def G_to_Coordinates(G,norm=False,ThD=False):
    '''
    Description
    ----------
    This function finds the Laplacian of a graph and the first two non-zero
    eigenvectors. If ThD is set to True, this finds the first three non-zero
    eigenvectors instead.
    
    If norm is set to True, the eigenvectors are normalized using the degree
    matrix
    
    The Laplacian and the eigenvectors are returned.
    
    Parameters
    ----------
    G       : Networkx graph
              Networkx graph of data.
    norm    : boolean, optional
              True returns degree-normalized data. The default is False.
    ThD     : boolean, optional
              True returns 3-Dimensional eigenvectors.
              The default is False which returns 2-Dimensional eigenvectors.

    Raises
    ------
    ValueError
        This error is raised if save is not boolean.

    Returns
    -------
    L       : numpy array
              Laplacian matrix of G.
    e_vec   : numpy array
              Eigenvectors of the data.

    '''
    
    # Get Laplacian from G
    L = nx.laplacian_matrix(G).toarray()
    if norm:
        D = np.diag(np.diag(L))
    else:
        D = np.eye(len(L))
    
    # Find eigenvalue
    if ThD:
        dim = 3
    else:
        dim = 2
    
    try:
        e_val, e_vec = la.eigh(a = L,
                               b = D,
                               subset_by_index=(1,dim))
    except TypeError:
        e_val, e_vec = la.eigh(a = L,
                               b = D,
                               eigvals=(1,dim))
    
    # Return Data Points
    return L, e_vec

def csv_to_Coordinates(file,thhld=0,norm=False,ThD=False):
    '''
    Description
    ----------
    Given a CSV file containing the weights of a dataset, this function creates
    and returns the graph G and the Laplacian matrix along with the first two
    non-trivial eigenvectors.
    
    Parameters
    ----------
    file    : string
              A CSV file containing the weights of edges connecting from a
              dataset.
    thhld   : float, optional
              Should be a value in [0,100]. This is used to exclude lighter
              weights
              from the graph. The default is 0.
    norm    : boolean, optional
              True returns degree-normalized data. The default is False.
    ThD     : boolean, optional
              True returns 3-Dimensional eigenvectors.
              The default is False which returns 2-Dimensional eigenvectors.

    Returns
    -------
    G           : Networkx graph
                  Networkx graph of data.
    laplacian   : numpy array
                  Laplacian Matrix of G.
    data        : numpy array
                  Eigenvectors of laplacian.

    '''
    
    # Get Graph
    G = csv_to_G(file,
                 thhld=thhld)
    
    # Eigen-Projection and Laplacian
    L, data = G_to_Coordinates(G,
                               norm=norm,
                               ThD=ThD)
    
    return G, L, data

def draw_coordinates(G,datapoints,colors='blue'):
    '''
    Description
    ----------
    Given a graph G and coordinates, this function will plot the graph and
    retain its edges. The vertices will be placed at the given coordinates.
    The vertices' colors may be changed.
    
    Parameters
    ----------
    G           : networkx graph
                  Graph of your data structure.
    datapoints  : numpy array
                  Input data of size n*2. n must be the number of vertices in G.

    Returns
    -------
    None.

    '''
    plt.clf()
    
    lines = np.array(G.edges()).T
    
    X1 = datapoints[:,0][lines]
    X2 = datapoints[:,1][lines]
    
    plt.plot(X1,X2,linewidth=.5,zorder=1)
    plt.scatter(datapoints[:,0],
                datapoints[:,1],
                edgecolors='black',
                zorder=2,
                c=colors,
                linewidths=0.5)

def coloring(arr):
    '''
    Description
    ----------
    Given the matrix of an image, this function converts it to a color array.
    The returned array may be used in draw_coordinates to color each vertex
    the same as the color of the original image.
    
    Parameters
    ----------
    arr : numpy array
        Must be an image array.

    Returns
    -------
    c_arr   : numpy array
              This is a scaled and reshaped copy of arr. This is suitable for
              colors in plots.

    '''
    
    try:
        r,c,dim = arr.shape[0],arr.shape[1],arr.shape[2]
    except ValueError:
        r,c = arr.shape[0],arr.shape[1]
        dim = 1
    
    c_arr = arr.reshape(r*c,dim)/255
    
    return c_arr

def spectral_clustering(infile,
                        dist=1,
                        c_alpha=1,
                        l_alpha=1,
                        norm=False,
                        ThD=False,
                        draw=False):
    '''
    Description
    ----------
    This function combines many functions to complete the spectral clustering
    process.
    
    This takes in an image, converts it to a matrix and finds weights relating
    pixels. The weights are saved as a CSV in ./Data. The weights are then used
    to make a networkx graph.
    
    The Laplacian and first two non-trivial eigenvectors are found. They are
    saved in ./Data. The graph is drawn if the parameter 'draw' is set to True.
    This drawing is saved in ./Data.
    
    Parameters
    ----------
    infile    : string
                Image file.
    dist      : integer, optional
                Maximum distance between data points to be considered adjacent.
                The default is 1.
    c_alpha   : integer, optional
                Scalar value which adjusts the norm of the color matrix.
                Accepts values greater than 0. The default value is 1.
    l_alpha   : integer, optional
                Scalar value which adjusts the norm of the location matrix.
                Accepts values greater than 0. The default value is 1.
    norm      : boolean, optional
                True returns degree-normalized data. The default is False.
    ThD       : boolean, optional
                True returns 3-Dimensional eigenvectors.
                The default is False which returns 2-Dimensional eigenvectors.

    Returns
    -------
    data       : numpy array
                 Eigenvectors of laplacian.
    r          : integer
                 Number of rows in grid.
    c          : integer
                 Number of columns in grid.
    G          : networkx graph
                 Networkx graph of data.
    L          : numpy array
                 Laplacian matrix of G.
    path       : string
                 Path location of the image data.
    img_mtx    : numpy array
                 An n*m*k array of the image. k is dependent on the color
                 palette of the image.

    '''
    path, r, c, img_mtx = img_to_csv(infile,
                                     dist=dist,
                                     c_alpha=c_alpha,
                                     l_alpha=l_alpha)
    
    G, L, data = csv_to_Coordinates(path,
                                    norm=norm,
                                    ThD=ThD)
    
    folder_path = os.path.split(path)[0]
    
    pd.DataFrame(data).to_csv('{}//eigval_norm={}_D={},Ca={},La={}.csv'.format(folder_path,norm,dist,c_alpha,l_alpha))
    print('Eigenvectors saved in {} as eigval_norm={}_D={},Ca={},La={}.csv'.format(folder_path,norm,dist,c_alpha,l_alpha))
    
    pd.DataFrame(L).to_csv('{}//Laplacian_norm={}_D={},Ca={},La={}.csv'.format(folder_path,norm,dist,c_alpha,l_alpha))
    print('Laplacian saved in {} as Laplacian_norm={}_D={},Ca={},La={}.csv'.format(folder_path,norm,dist,c_alpha,l_alpha))
    
    if norm:
        dnorm = "  Normalized"
    else:
        dnorm = ""
    
    if draw:
        color_val = coloring(img_mtx)
        draw_coordinates(G,data,colors=color_val)
        plt.title("Max Distance = {}  α  = 1/{}  α  = 1/{}{}".format(dist, c_alpha, l_alpha, dnorm))
        plt.savefig('{}//eig_graph_norm={}_D={},Ca={},La={}.png'.format(folder_path,norm,dist,c_alpha,l_alpha))
        print('Plot saved in {} as eig_graph_norm={}_D={},Ca={},La={}.csv'.format(folder_path,norm,dist,c_alpha,l_alpha))
    return data, r, c, G, L, path, img_mtx

def plotting(arr,r,c):
    '''
    Description
    ----------
    Given an array with r*c entries, this function shows an r x c heatmap of
    the array.
    
    Parameters
    ----------
    arr     : numpy array
              Data to be plotted.
    r, c    : integer
              Dimension of the data.

    Raises
    ------
    ValueError
        The product of r*c must be the size of arr.

    '''
    plt.clf()
    
    try:
        arr = arr.reshape(r,c)
    except ValueError:
        raise ValueError("The values for r and c are inappropriate.")
    
    plt.imshow(arr,cmap="Greys")