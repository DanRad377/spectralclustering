import spectral_clustering as spcl
from PIL import Image

image = 'Images//tree.png'

img = Image.open(image)

img.show()

data, r, c, G, L, path, img_mtx = spcl.spectral_clustering(image,
                                                           draw=True,
                                                           dist=2,
                                                           c_alpha=1000,
                                                           l_alpha=4,
                                                           norm=True)

