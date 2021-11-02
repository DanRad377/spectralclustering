# Functions in this Package

This package contains a variety of functions which all contribute to use spectral clustering on an image. Here are a few notable ones.

## img_to_mtx

This function converts an image to a numpy array. It requires a file as an input parameter.

**Example**: `arr = img_to_mtx(file = "./image.png")`

## color_sim

The color similarity function relates each pixel of the image based on their color. Input an image's array from **img_to_mtx** and set the alpha value.

The *alpha* value is a way for the user to scale this function. It defaults as 1. This value is must be positive.

This function returns a color weight array along with the row and column sizes of the array.

**Example**: `color_w, r, c = color_sim(arr,alpha=1)`

## loc_sim

The location similarity function relates each pixel of the image based on their distance from one another. Pixels outside of a certain radius are considered unrelated.

To use this function, you need the dimensions of the image, which color_sim does provide. The maximum radius is the variable *dist*. The default radius is 1 and must be at least 1.

There is also an *alpha* parameter which scales this function. The default of which is 1. This value must be positive.

This function returns a location weight array.

**Example**: `location_w = loc_sim(r,c,dist=1,alpha=1)`

## img_to_csv

This function performs all of the previous functions along with a few others to create a csv file which stores an array of a weight matrix representing an image file.

This function requires a value *infile* which should be a local image file.

We also see *dist* which is the maximum distance where pixels should be considered related.

*c_alpha* and *l_alpha* are the color and location alpha values mentioned earlier.

## csv_to_Coordinates

This function makes use of the weights csv file which was created in **img_to_csv**. It creates a graph in networkx from the weights. Then it finds the graph's Laplacian and the non-zero eigenvectors.

*file* is the weight csv file.

The boolean value *norm* is false by default. If true, this function will find the degree-normalized eigenvectors.

The boolean value *ThD* is false by default. If true, the eigenvectors will be three-dimensional. They will be two-dimensional otherwise.

The parameter *thhld* is a threshold. It can be any value between 0 and 100 inclusively. The default is 0. This parameter will disassociate vertices (i.e. remove edges from the graph). The number is a percentage of the greatest weight. If thhld is set to 50, then the function will disassociate vertices if the edge's weight is less than 50% of the greatest weight among edges.

This function outputs a networkx graph, a Laplacian matrix and a set of vectors both as numpy arrays.

**Example**: `G, L, vec = csv_to_Coordinates(file,thhld=0,norm=False,ThD=False)`

## spectral_clustering

This function is the combination of all the previous ones. The parameters will therefore be redundant.
1. *infile* is the image file.
1. *dist* is the maximum distance between pixels.
1. *c_alpha* and *l_alpha* are the color and location alpha values respectively.
1. *norm* is a boolean selecting if the eigenvectors are degree-normalized.
1. *ThD* is a boolean that makes the eigenvectors 3-dimensional if set to true. They are 2-dimensional otherwise.
1. *draw* is a boolean that draws the repositioned graph if set to true. It also saves the graph as a png file.