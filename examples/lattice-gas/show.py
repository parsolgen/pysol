import matplotlib.pyplot as plt
import matplotlib.image as img

# reading png image file
im = img.imread('out.png')

# show image
plt.imshow(im)