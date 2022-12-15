def unzip_data(filename, dest, del_MACOSX_file=True):
    '''
    Unzips the file ina given destination.

    Args:
        filename (str): name of the file to be unzippes
        dest (str): destination where file will be unzipped
        del_MACOSX_file (bool): do you want MACOSX file to be deleted?

    Returns
        Unzipped file in a given directory
    '''

    import zipfile
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall(dest)
    zip_ref.close()

    import shutil
    shutil.rmtree('data/__MACOSX')


def walk_through_dir(dir_name):
    '''
    Walks through a diven directory. A directory should contain of 
    folders and files within these folders: path -> directories -> files

    Args: 
        dir_name (str): name of a directory to be walked through. 

    Returns:
        Formated string: 'There are {len(dirnames)} directories and {len(filenames)} images in "{dirpath}".'
    '''
    import os

    for dirpath, dirnames, filenames in os.walk(dir_name):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in "{dirpath}".')


def view_random_image(target_dir, target_class):
    '''
    Prints a random image in target_class from target_dir.
    '''
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random
    import os

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


def plot_loss_curves(history, metric_name='accuracy'):
    '''
    Returns sepatrate loss curves for training and validation metrics.

    Args:
        history: TensorFlow history object.
        metric_name: metric used during fitting a model, default: accuracy

    Returns:
        Plots of training/validation loss and metric
    '''
    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    metric = history.history[metric_name]
    val_metric = history.history['val_'+metric_name]

    epochs = range(len(history.history['loss'])) # how many epochs did we run for

    # Plot loss
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metric, label='training_'+metric_name)
    plt.plot(epochs, val_metric, label='val_'+metric_name)
    plt.title(metric_name)
    plt.xlabel('epochs')
    plt.legend()


def create_tensorboad_callback(dir_name, experiment_name):
    '''
    Creates a TensorBoard callback.

    Args:
        dir_name (str): A directory where TensorBoard output will be saved.
        experiment_name (str): name of the experiment.

    Returns:
        TensorBoard callback stored in a dir_name directory.
    '''
    import datetime
    import tensorflow as tf

    # Create a log directory
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Set the TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Print output
    print(f'Saving TensorBoad log files to {log_dir}')

    return tensorboard_callback


def create_model(model_url, trainable=False, layer_name='feature_extraction_layer', 
                 image_shape=(224, 224), clasification_type='multiclass', num_classes=10):
    '''
    Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
        model_url (str): A TensorFlow Hub feature extraction URL.
        trainable (bool): Do you want to reset already learned features?
            Default False.
        layer_name (str): name of the feature_extractor layer,
            default: "feature_extraction_layer"
        image_shape (int, int): shape of the image (height, width),
            default: (224, 224)
        classification_type (str): "multiclass" for multiclass classification,
            "binary" for binary classification
        num_classes (int): Number of output neurons in the output layer,
            should be equal to number of target classes, default 10.

    Returns:
        An uncompiled Keras Sequential model with model_url as feature extractor
        layer and Dense output layer with num_classes output neurons.
    '''
    import tensorflow as tf
    import tensorflow_hub as hub

    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=trainable, # freeze the already learned patterns
                                             name=layer_name, 
                                             input_shape=image_shape+(3,)) 

    # Set output layer activation function based on clasification_type:
    if clasification_type == 'multiclass': 
        activation_function = 'softmax' 
    elif clasification_type == 'binary':
        activation_function = 'sigmoid' 
    else:
        raise ValueError('clasification_function must be "multiclass" or "binary"')

    # Create a model
    model = tf.keras.models.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes, activation=activation_function)
    ])

    return model

# ---

def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow History objects
    """
    import matplotlib.pyplot as plt

    # Get original history measurements
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']

    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']   

    # Combine original history metrics with new_history metrics
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']

    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    # Make a plot for accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training accuracy')
    plt.plot(total_val_acc, label='Validation accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Make a plot for loss
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training loss')
    plt.plot(total_val_loss, label='Validation loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), color_bar=False, text_size=15):

    import itertools
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize our confusion matrix
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # plot the values of a 2D matrix or array as color-coded image.
    if color_bar == True:
        fig.colorbar(cax) # colorbar

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axis 
    ax.set(title='Confusion Matrix',
        xlabel='Predicted Label',
        ylabel='True Label',
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

    # Set x-axis labels to the bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size+5)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i , f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
                horizontalalignment='center',
                color='white' if cm[i, j] > threshold else 'black',
                size=text_size)