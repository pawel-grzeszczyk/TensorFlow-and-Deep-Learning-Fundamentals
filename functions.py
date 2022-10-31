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