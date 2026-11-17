# Point registration and image mosaics

Python 3.11 and NumPy. Run `python demo.py` from this folder.
`align.fit` currently fits a translation, and `align.project` applies a matrix.
`mosaic.compose` accepts external matrices and currently uses nearest samples
and independently selected sources. Fixtures are synthetic grayscale values.
The masks mark valid observations. Matrix coordinates map source to canvas;
pixel coordinates use x to the right and y downward, with integer centers.
