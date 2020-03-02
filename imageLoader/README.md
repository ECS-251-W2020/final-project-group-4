# image_to_numpy

Load an image file, for exaple, `*.jpg`, as a numpy arrary

The image is automatically rotated into the correct orientation if the image contains Exif orientation metadata.
Otherwise, it is just a normal load.

After loading, you can pass the array to some other Python library or function which works with Numpy arrary.

## Note about Exif Orientation data?

Most images captured by cell phones and consumer cameras aren't stored on disk in the same orientation they appear on screen. Exif Orientation data tells the program which way the image needs to be rotated to display correctly. Not handling Exif Orientation is a common source of bugs in Computer Vision and Machine Learning applications.


## Guide

 ```python
import image_to_numpy

img = image_to_numpy.load_image_file("my_file.jpg")
```

By default, the image array is returned as a numpy array with 3-channels of 8-bit RGB data.

You can control the output format by checking the `mode` parameter in `load_image_file` function interface:

You can show your `img` by matpoltlib very quickly:

```python
import matplotlib.pyplot as plt
import image_to_numpy

img = image_to_numpy.load_image_file("my_file.jpg")

plt.imshow(img)
plt.show()
```
