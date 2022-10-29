import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import os

def view_random_image(target_dir, target_class):
    '''
    Prints a random image in target_class from target_dir.
    '''
    # Setup the target directory
    target_folder = target_dir + '/' + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)

    # Read in the image and plot it
    img = mpimg.imread(target_folder + '/' + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off');

    # Show the shape of the image
    print(f'Image shape: {img.shape}') 

    return img