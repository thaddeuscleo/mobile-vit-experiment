from mlflow import log_metric, log_params, log_artifacts

import tensorflow as tf

from keras.applications import imagenet_utils
from keras import layers
from tensorflow import keras

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.

def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC
def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features

def create_mobilevit(num_classes=5):
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


mobilevit_xxs = create_mobilevit()
mobilevit_xxs.summary()

batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 3

def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            # print(image_size)
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image[0], size=(image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
            image = image[0]
            label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.batch(batch_size).prefetch(auto)

train_dataset_dir = "./Banana_Condensed_Three_Cat"

train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dataset_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    subset="both",
    seed=42
)

# import tensorflow_datasets as tfds
num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f"Number of training examples: {num_train}")
print(f"Number of validation examples: {num_val}")

def get_class_count_from_dataset(dataset):
  labels = []
  for _, y in dataset:
    labels.append(np.argmax(y, axis = -1))
  labels = np.concatenate(labels, axis = 0)
  _, counts = np.unique(labels, return_counts=True)
  return pd.DataFrame(counts, columns=["count"], index=dataset.class_names)

train_df = get_class_count_from_dataset(train_dataset)
val_df = get_class_count_from_dataset(val_dataset)

train_weight = {}
for idx, cnt in enumerate(train_df['count']):
    print(f"{train_dataset.class_names[idx]}: {cnt}")
    train_weight[idx] =  (cnt) / (12.0 * 14762)


# Hyperparameter
learning_rate = 0.00002
label_smoothing_factor = 0.1
epochs = 70

log_params({
    "learning_rate": learning_rate,
    "label_smoothing_factor": label_smoothing_factor,
    "epochs": epochs
})

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)

def recall_m(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+ keras.backend.epsilon()))

def run_experiment(epochs=epochs):
    mobilevit_xxs = create_mobilevit(num_classes=num_classes)
    mobilevit_xxs.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy", recall_m, precision_m, f1_m])

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    mobilevit_xxs.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        class_weight=train_weight
    )

    mobilevit_xxs.load_weights(checkpoint_filepath)
    _, accuracy, recall_val, precision_val, f1_val = mobilevit_xxs.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    return mobilevit_xxs


mobilevit_xxs = run_experiment()

# !rm -rf "/content/saved_model"
# !mkdir -p saved_model

# Serialize the model as a SavedModel.
# mobilevit_xxs.save("saved_model/mobilevit_xxs")
import pickle

# Save the model weights
mobilevit_xxs.save_weights("./saved_model/mobilevit_xxs_weights.h5")

# Save the model architecture
json_config = mobilevit_xxs.get_config()
f = open('./saved_model/mobilevit_xxs_architecrure.pickle', "wb")
f.write(pickle.dumps(json_config))
f.close()

# save the models labels
CLASS_NAMES = val_dataset.class_names
f = open('./saved_model/labels.pickle', "wb")
f.write(pickle.dumps(CLASS_NAMES))
f.close()

log_artifacts("./saved_model")

def get_confusion_matrix(model, validation_dataset):
    all_predictions = np.array([])
    all_labels = np.array([])
    for i, batch in enumerate(validation_dataset):
        x_batch, y_batch = batch
        predictions = model.predict(x_batch)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        labels = np.argmax(y_batch, axis = 1)
        all_labels = np.concatenate([all_labels, labels])

    return tf.math.confusion_matrix(all_predictions, all_labels)


conffusion_mtx = get_confusion_matrix(mobilevit_xxs, val_dataset)

conffusion_mtx_df = pd.DataFrame(conffusion_mtx, index=val_dataset.class_names, columns=val_dataset.class_names)
plt.figure(figsize=(10,7))
sns.heatmap(conffusion_mtx_df, annot=True)

loss, accuracy, f1_score, precision, recall = mobilevit_xxs.evaluate(val_dataset.take(1), verbose=0)
print(f"loss: {loss}")
print(f"accuracy: {(accuracy * 100):.2f}%")
print(f"f1_score: {f1_score}")
print(f"precision: {precision}")
print(f"recall: {recall}")

log_metric("loss", loss)
log_metric("accuracy", accuracy * 100)
log_metric("f1_score", f1_score)
log_metric("precision", precision)
log_metric("recall", recall)

