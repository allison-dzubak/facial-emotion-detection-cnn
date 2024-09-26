# Import the libraries
import os
import shutil
from PIL import Image
import base64
from IPython.display import display, HTML
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


def count_jpg_files(directory):
    """
    Counts the number of .jpg files in the specified directory and all its subdirectories.

    Args:
    directory: The root directory to search for .jpg files. This function traverses all
               subdirectories under this root directory.

    Returns:
    An integer representing the total number of .jpg files found in the specified directory
    and its subdirectories.
    """
    jpg_count = 0
    for root, _, files in os.walk(directory):
        jpg_count += len(glob.glob(os.path.join(root, '*.jpg')))
    return jpg_count


def print_number_of_files(directory):
    """
    Prints the number of .jpg files in each subdirectory of the specified directory.

    Args:
    directory: The root directory to search for .jpg files. This function traverses all
               subdirectories under this root directory.
    """
    for subdir, _, _ in os.walk(directory):
        jpg_count = count_jpg_files(subdir)
        print(f"Number of .jpg files in {subdir}: {jpg_count}")


def show_pie_charts(directory):
    """
    Generates pie charts to visualize the distribution of images across different emotions
    in the train, validation, and test datasets.

    Args:
    directory: The root directory containing 'train', 'validation', and 'test' subdirectories,
               each with subdirectories for different emotions ('happy', 'sad', 'surprise', 'neutral').
    """
    # Count of images for each emotion in train, validation, and test
    emotions = ['happy', 'sad', 'surprise', 'neutral']
    train_counts = [count_jpg_files(os.path.join(directory, 'train', emotion)) for emotion in emotions]
    validation_counts = [count_jpg_files(os.path.join(directory, 'validation', emotion)) for emotion in emotions]
    test_counts = [count_jpg_files(os.path.join(directory, 'test', emotion)) for emotion in emotions]

    # Calculate total number of images in train, validation, and test sets
    total_train_images = sum(train_counts)
    total_validation_images = sum(validation_counts)
    total_test_images = sum(test_counts)

    # Calculate proportions
    total_images = total_train_images + total_validation_images + total_test_images
    train_proportions = np.array(train_counts) / total_images
    validation_proportions = np.array(validation_counts) / total_images
    test_proportions = np.array(test_counts) / total_images

    # Plot side-by-side pie charts
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].pie(train_proportions, labels=emotions, autopct='%1.1f%%', startangle=90)
    axs[0].set_title('Train Set')

    axs[1].pie(validation_proportions, labels=emotions, autopct='%1.1f%%', startangle=90)
    axs[1].set_title('Validation Set')

    axs[2].pie(test_proportions, labels=emotions, autopct='%1.1f%%', startangle=90)
    axs[2].set_title('Test Set')

    plt.show()

    # Add line
    print("\n" + "="*55)

    # Print the percentage of total images in each set
    print(f"\nNumber Total Images in Train Set: {total_train_images} ({total_train_images / total_images * 100:.2f}%)")
    print(f"Number of Total Images in Validation Set: {total_validation_images} ({total_validation_images / total_images * 100:.2f}%)")
    print(f"Number of Total Images in Test Set: {total_test_images} ({total_test_images / total_images * 100:.2f}%)")
    print(f"Number of Total Images: {total_images}")


def new_session():
    """
    Clears the current TensorFlow session to free up memory and reset the state.
    Sets the seed for random number generators to ensure reproducibility.
    """
    # Clear the TensorFlow backend session
    backend.clear_session()

    # Set the seed for NumPy random number generator
    np.random.seed(1991921)

    # Set the seed for Python's built-in random number generator
    random.seed(1991921)

    # Set the seed for TensorFlow random number generator
    tf.random.set_seed(1991921)


def transfer_data(preprocessed_data_directory, temp_data_dir):
    """
    Transfers data from the preprocessed directory to the temporary Colab filesystem if it's not already there.

    Args:
    preprocessed_data_directory (str): Path to the preprocessed data directory.
    base_dir (str): Path to the base directory in the Colab filesystem where the data should be copied.

    Returns:
    None
    """
    # Check if the base directory exists
    if not os.path.exists(temp_data_dir):
        print(f"{temp_data_dir_dir} does not exist, copying data...")
        shutil.copytree(preprocessed_data_directory, temp_data_dir, dirs_exist_ok=True)
        print(f"Data copied to {temp_data_dir}")
    else:
        # Check if the directory is empty
        if not os.listdir(temp_data_dir):
            print(f"{temp_data_dir} is empty, copying data...")
            shutil.copytree(preprocessed_data_directory, temp_data_dir, dirs_exist_ok=True)
            print(f"Data copied to {temp_data_dir}")
        else:
            print(f"Data already exists at {temp_data_dir}, no need to transfer again.")


def adjust_contrast_and_gamma(img):
    """
    Applies contrast adjustment followed by gamma correction to an image.

    - First, the image's intensity is rescaled to enhance contrast.
    - Then, gamma correction is applied using a random gamma value between 0.5 and 2.0.

    Args:
    img: Input image to be adjusted (assumed to be in NumPy array format).

    Returns:
    gamma_corrected_img: The image after contrast adjustment and gamma correction.
    """
    # Adjust contrast by rescaling image intensity
    adjusted_img = exposure.rescale_intensity(img)

    # Apply random gamma correction (gamma value between 0.5 and 2.0)
    gamma = np.random.uniform(0.5, 2.0)
    gamma_corrected_img = exposure.adjust_gamma(adjusted_img, gamma)

    return gamma_corrected_img


def get_train_datagen(augmentation_level):
    """
    Returns a train data generator with the specified level of data augmentation.

    Args:
    augmentation_level (str): Specifies the level of augmentation.
                             Options: 'none', 'low', 'mid', 'high'

    Returns:
    ImageDataGenerator: The data generator with the specified level of augmentation.
    """
    # Base rescaling and common augmentations
    base_augmentation = {
        'rescale': 1. / 255.,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True
    }

    if augmentation_level == 'none':
        return ImageDataGenerator(rescale=1. / 255.)

    elif augmentation_level == 'low':
        return ImageDataGenerator(**base_augmentation)

    elif augmentation_level == 'mid':
        return ImageDataGenerator(
            **base_augmentation,
            rotation_range=15
        )

    elif augmentation_level == 'high':
        return ImageDataGenerator(
            **base_augmentation,
            rotation_range=15,
            preprocessing_function=adjust_contrast_and_gamma
        )

    else:
        raise ValueError("Invalid augmentation level. Choose from 'none', 'low', 'mid', or 'high'.")


def flow_data(train_augmentation_level, train_dir, validation_dir, batch_size):
    """
    Sets options for data flow using specified data augmentation levels for training.
    Validation data is rescaled without augmentation.

    Args:
    train_augmentation_level (str): Augmentation level for training data.
                                    Options: 'none', 'low', 'mid', 'high'
    train_dir (str): Directory for training data.
    validation_dir (str): Directory for validation data.
    batch_size (int): Batch size for the data generators.

    Returns:
    train_generator, validation_generator: Image data generators for training and validation.
    """
    # Get train data generator for the specified augmentation level
    train_datagen = get_train_datagen(train_augmentation_level)

    # Validation data generator (no augmentation, only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1. / 255.)

    # Flow images from directories
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        class_mode='categorical',
                                                        target_size=(224, 224),
                                                        batch_size=batch_size)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  class_mode='categorical',
                                                                  target_size=(224, 224),
                                                                  batch_size=batch_size)

    return train_generator, validation_generator


def set_model_bottom(architecture_choice, num_unfreeze):
    """
    Sets the base model architecture based on the specified choice. Supports VGG16, ResNet50, and EfficientNetB0.
    Sets the convolution blocks as frozen / trainable based on num_unfreeze

    Args:
    architecture_choice (str): The name of the architecture to use. Options are 'VGG16', 'ResNet50', or 'EfficientNetB0'.
    num_unfreeze (int or str): Specified the number of convolution blocks to unfreeze
                                Options: 0, 1, 2, 3, 'all'

    Returns:
    model_bottom: Pre-trained model without the top fully connected layers.

    Raises:
    ValueError: If an invalid architecture choice is provided.
    """
    # Load the specified model architecture without the top (fully connected) layers
    if architecture_choice == 'VGG16':
        # Load VGG16 model with pre-trained ImageNet weights, excluding the fully connected layers
        model_bottom = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Handle layer freezing/unfreezing logic based on the 'num_unfreeze' argument
        if num_unfreeze == 0:  # Freeze all layers
            for layer in model_bottom.layers:
                layer.trainable = False
        elif num_unfreeze == 1:  # Unfreeze only the top convolution block
            for layer in model_bottom.layers:
                if layer.name.startswith('block5'):
                    break  # Start unfreezing from block5 onward
                layer.trainable = False
        elif num_unfreeze == 2:  # Unfreeze the top two convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('block4'):
                    break  # Start unfreezing from block4 onward
                layer.trainable = False
        elif num_unfreeze == 3:  # Unfreeze the top three convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('block3'):
                    break  # Start unfreezing from block3 onward
                layer.trainable = False
        elif num_unfreeze == 'all':  # Unfreeze all layers
            for layer in model_bottom.layers:
                layer.trainable = True

    elif architecture_choice == 'ResNet50V2':
        # Load ResNet50V2 model with pre-trained ImageNet weights, excluding the fully connected layers
        model_bottom = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Handle layer freezing/unfreezing logic based on the 'num_unfreeze' argument
        if num_unfreeze == 0:  # Freeze all layers
            for layer in model_bottom.layers:
                layer.trainable = False
        elif num_unfreeze == 1:  # Unfreeze only the top convolution block
            for layer in model_bottom.layers:
                if layer.name.startswith('conv5_block'):
                    break  # Start unfreezing from block5 onward
                layer.trainable = False
        elif num_unfreeze == 2:  # Unfreeze the top two convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('conv4_block'):
                    break  # Start unfreezing from block4 onward
                layer.trainable = False
        elif num_unfreeze == 3:  # Unfreeze the top three convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('conv3_block'):
                    break  # Start unfreezing from block3 onward
                layer.trainable = False
        elif num_unfreeze == 'all':  # Unfreeze all layers
            for layer in model_bottom.layers:
                layer.trainable = True

    elif architecture_choice == 'EfficientNetB0':
        # Load EfficientNetB0 model with pre-trained ImageNet weights, excluding the fully connected layers
        model_bottom = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Handle layer freezing/unfreezing logic based on the 'num_unfreeze' argument
        if num_unfreeze == 0:  # Freeze all layers
            for layer in model_bottom.layers:
                layer.trainable = False
        elif num_unfreeze == 1:  # Unfreeze only the top convolution block
            for layer in model_bottom.layers:
                if layer.name.startswith('block7a'):
                    break  # Start unfreezing from block7a onward
                layer.trainable = False
        elif num_unfreeze == 2:  # Unfreeze the top two convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('block6a'):
                    break  # Start unfreezing from block6a onward
                layer.trainable = False
        elif num_unfreeze == 3:  # Unfreeze the top three convolution blocks
            for layer in model_bottom.layers:
                if layer.name.startswith('block5a'):
                    break  # Start unfreezing from block5a onward
                layer.trainable = False
        elif num_unfreeze == 'all':  # Unfreeze all layers
            for layer in model_bottom.layers:
                layer.trainable = True

    else:
        # Raise an error if the architecture choice is invalid
        raise ValueError("Invalid architecture choice. Please choose from 'VGG16', 'ResNet50V2', or 'EfficientNetB0'.")

    return model_bottom


def build_top_layers(architecture_choice, model_bottom, add_dense):
    """
    Sets the base model architecture top layers based on the specified choices. Supports VGG16, ResNet50V2, and EfficientNetB0.

    Args:
    architecture_choice (str): The name of the architecture to use. Options are 'VGG16', 'ResNet50V2', or 'EfficientNetB0'.
    model_bottom (Model): The pre-trained base model whose output will be used as input for the top layers.
    add_dense (bool): Whether or not to add an additional dense layer to the top of the model.

    Returns:
    model (Model): The final model with the base and top layers combined.

    Raises:
    ValueError: If an invalid architecture choice is provided.
    """
    # Start from the output of the pre-trained model
    x = model_bottom.output

    if architecture_choice == 'VGG16':
        # Build the top layers of the model
        x = Flatten()(x)

        # Add two fully connected (Dense) layers with 4096 units
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)

        # Optionally add an extra Dense layer if 'add_dense' is True
        if add_dense:
            x = Dense(4096, activation='relu')(x)

    elif architecture_choice == 'ResNet50V2':
        # Build the top layers of the model
        x = GlobalAveragePooling2D()(x)

        # Optionally add an extra Dense layer if 'add_dense' is True
        if add_dense:
            x = Dense(4096, activation='relu')(x)

    elif architecture_choice == 'EfficientNetB0':
        # Build the top layers of the model
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        # Optionally add an extra Dense layer if 'add_dense' is True
        if add_dense:
            x = Dense(1280, activation='relu')(x)

    else:
        # Raise an error if the architecture choice is invalid
        raise ValueError("Invalid architecture choice. Please choose from 'VGG16', 'ResNet50V2', or 'EfficientNetB0'.")

    # Add the final output layer with softmax activation (for 4 classes)
    x = Dense(4, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=model_bottom.input, outputs=x)

    return model


def set_model_structure(architecture_choice, num_unfreeze, add_dense):
    """
    Creates the full model architecture by combining a pre-trained base model with custom top layers.

    Args:
    architecture_choice (str): The name of the architecture to use. Options are 'VGG16', 'ResNet50V2', or 'EfficientNetB0'.
    num_unfreeze (int): Number of layers to unfreeze for training in the base model.
    add_dense (bool): Whether or not to add an additional dense layer to the top of the model.

    Returns:
    model (Model): The complete model combining the pre-trained base and custom top layers.
    """
    # Get the pre-trained model bottom with specified layers frozen / trainable
    model_bottom = set_model_bottom(architecture_choice, num_unfreeze)

    # Build the top layers of the model based on architecture choice
    model = build_top_layers(architecture_choice, model_bottom, add_dense)

    return model


def set_callbacks(checkpoint_filepath):
    """
    Sets up the ModelCheckpoint and EarlyStopping callbacks for model training.

    Args:
    checkpoint_filepath (str): The file path where the model's weights will be saved.

    Returns:
    Tuple of callbacks:
    - ModelCheckpoint: Saves the model's weights when validation accuracy improves.
    - EarlyStopping: Stops training if validation accuracy doesn't improve after a certain number of epochs.
    """
    # ModelCheckpoint: Saves the best model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          save_weights_only=True)

    # EarlyStopping: Stops training if validation accuracy doesn't improve for 2 consecutive epochs
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=2)

    # Return both callbacks
    return checkpoint_callback, early_stopping_callback


def plot_accuracy_per_epoch(training_history):
    """
    Plots the accuracy of the model over each epoch during training.

    Args:
    training_history: A history object or dictionary-like object returned by the
                      Keras model's fit method. It should contain accuracy and
                      validation accuracy for each epoch.

    Returns:
    Displays a plot showing training and validation accuracy per epoch.
    """
    # Convert the training history to a DataFrame for easier plotting
    history_df = pd.DataFrame(training_history)

    # Plot training and validation accuracy
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')

    # Add a legend in the upper left corner
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()


def show_classification_performance(model, validation_dir, num_per_row=6):
    """
    Evaluates the performance of a trained model on a validation dataset by generating:
    - Confusion matrix
    - Classification report
    - Visualization of misclassified images

    Args:
    model: Trained Keras model for which performance needs to be evaluated.
    validation_dir: Path to the validation dataset directory.
    num_per_row: Maximum number of misclassified images to display per row (default is 6).

    Returns:
    Displays confusion matrix, classification report, and HTML visualization of misclassified images.
    """
    # Data generator for validation data (no augmentation, only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1. / 255.)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  class_mode='categorical',
                                                                  target_size=(224, 224),
                                                                  shuffle=False)

    # Predict the classes for the validation data
    y_pred = model.predict(validation_generator, verbose=1)
    y_true = validation_generator.classes

    # Get class labels from the generator
    class_labels = list(validation_generator.class_indices.keys())

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred.argmax(axis=1))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print classification report
    print('\nClassification Report:\n', classification_report(y_true, y_pred.argmax(axis=1), target_names=class_labels))

    # Identify misclassified samples
    misclassified_indices = np.where(y_pred.argmax(axis=1) != y_true)[0]
    num_misclassified = len(misclassified_indices)

    if num_misclassified == 0:
        print("No misclassified images found!")
        return

    # Group misclassified images by the type of misclassification (True label vs. Predicted label)
    misclassified_images = {}
    for idx in misclassified_indices:
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx].argmax()]
        misclassification_type = f'True={true_label}, Predicted={pred_label}'

        # Append image paths to the corresponding misclassification type
        if misclassification_type not in misclassified_images:
            misclassified_images[misclassification_type] = []
        misclassified_images[misclassification_type].append(validation_generator.filepaths[idx])

    # Sort the misclassified images by the number of images in each type
    sorted_misclassified_images = dict(
        sorted(misclassified_images.items(), key=lambda item: len(item[1]), reverse=True))

    # Construct HTML to display misclassified images
    display_str = ''
    for misclassification_type, images in sorted_misclassified_images.items():
        display_str += f'<p style="font-weight:bold;">Misclassification type: {misclassification_type}</p>'
        display_str += '<div style="display:flex; flex-wrap: wrap;">'

        # Display images (up to num_per_row)
        for img_path in images[:num_per_row]:
            # Read and encode the image as base64
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("ascii")

            # Construct the HTML to display the image
            img_html = f'<div style="margin: 5px; text-align: center;">'
            img_html += f'<img src="data:image/png;base64,{img_data}" style="width: 180px; height: auto;"/><br>'
            img_html += '</div>'
            display_str += img_html

        display_str += '</div><br>'

    # Display the HTML with the misclassified images
    display(HTML(display_str))


def run_training_workflow(main_directory, temp_directory, training_parameters):
    """
    Runs the training workflow for a model given specified training parameters. This includes setting up the training environment,
    accessing the preprocessed data, configuring the model, training the model, and evaluating its performance.

    Args:
    main_directory (str): Path to the main directory containing preprocessed data and other resources.
    temp_directory (str): Path to the temporary directory where data will be transferred for faster access during training.
    training_parameters (dict): Dictionary containing the parameters for model training. The keys must include:
        - 'model_name': Name of the model (used for checkpoint filenames).
        - 'train_data_aug': Data augmentation level ('none', 'low', 'mid', 'high').
        - 'architecture_choice': Model architecture choice ('VGG16', 'ResNet50V2', 'EfficientNetB0').
        - 'num_unfreeze': Number of layers to unfreeze in the pre-trained model.
        - 'add_dense': Boolean indicating whether to add an additional Dense layer.
        - 'learning_rate': Learning rate for the optimizer.
        - 'batch_size': Batch size for training.
        - 'class_weight_balance': Boolean indicating whether to apply class weights to balance the dataset.
        - 'epochs': Number of training epochs.

    Raises:
    ValueError: If any of the required keys are missing in the training_parameters dictionary.
    """
    # Ensure all required parameters are provided
    required_keys = ['model_name', 'train_data_aug', 'architecture_choice', 'num_unfreeze',
                     'add_dense', 'learning_rate', 'batch_size', 'class_weight_balance', 'epochs']

    for key in required_keys:
        if key not in training_parameters:
            raise ValueError(f"Missing required parameter: {key}")

    # Unpack parameters from the dictionary
    model_name = training_parameters['model_name']
    train_data_aug = training_parameters['train_data_aug']
    architecture_choice = training_parameters['architecture_choice']
    num_unfreeze = training_parameters['num_unfreeze']
    add_dense = training_parameters['add_dense']
    learning_rate = training_parameters['learning_rate']
    batch_size = training_parameters['batch_size']
    class_weight_balance = training_parameters['class_weight_balance']
    epochs = training_parameters['epochs']

    # Specify path to training data in temporary Colab filesystem for faster access during training
    temp_data_dir = os.path.join(temp_directory, 'Unique_Images')
    os.makedirs(temp_data_dir, exist_ok=True)

    # Location of training and validation data on temporary Colab filesystem
    train_dir = os.path.join(temp_data_dir, 'train')
    validation_dir = os.path.join(temp_data_dir, 'validation')

    # Specify path to preprocessed data
    preprocessed_data_directory = os.path.join(main_directory, 'data/unique_images')

    # Clear session and fix the seed for random number generators
    new_session()

    # Define the checkpoint filepath
    checkpoint_filepath = os.path.join(temp_directory, f'{model_name}_checkpoint.weights.h5')

    # Transfer data to temporary Colab filesystem (if it's not already there)
    transfer_data(preprocessed_data_directory, temp_data_dir)

    # Flow images with designated level of data augmentation ('none', 'low', 'mid', 'high')
    train_generator, validation_generator = flow_data(train_data_aug, train_dir, validation_dir, batch_size)

    # Set the model structure
    model = set_model_structure(architecture_choice, num_unfreeze, add_dense)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define the callbacks
    checkpoint_callback, early_stopping_callback = set_callbacks(checkpoint_filepath)

    # Fit the model
    print("Start model training...")
    if class_weight_balance:
        # Calculate class weights for training and validation datasets
        train_class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes),
                                                   y=train_generator.classes)

        # Convert class weights to dictionaries
        train_class_weights_dict = dict(enumerate(train_class_weights))

        model_training = model.fit(train_generator,
                                   validation_data=validation_generator,
                                   class_weight=train_class_weights_dict,
                                   validation_steps=validation_generator.samples // validation_generator.batch_size,
                                   callbacks=[checkpoint_callback, early_stopping_callback],
                                   epochs=epochs)
    else:
        model_training = model.fit(train_generator,
                                   validation_data=validation_generator,
                                   callbacks=[checkpoint_callback, early_stopping_callback],
                                   epochs=epochs)

    # Show training and validation accuracy
    print("Show training and validation accuracies")
    plot_accuracy_per_epoch(model_training.history)

    # Show confusion matrix, classification report, and visualize misclassified images
    print("Show classification performance")
    show_classification_performance(model, validation_dir)
