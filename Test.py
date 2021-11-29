#%% Dataset
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# %% Build Model
model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)


# %% 產生雜訊

def gen_noise(dim=28, nex = 1):
    input_img_data = tf.random.uniform((nex, dim, dim, 1))
    return tf.Variable(tf.cast(input_img_data, tf.float32))

#%%
import matplotlib.pyplot as plt
import numpy as np

def display_features(output_images, filter_titles=None, ncols=10, zoom = 5, sat_exp=2.0, val_exp = 1.0):
    nrows = int(np.ceil(len(output_images[-1]) / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5,nrows*5))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace = 0.1, wspace = 0.01)
    for axi, ax in enumerate(axs.flatten()):
        if filter_titles is not None:
            if axi < len(filter_titles):
                ax.set_title(filter_titles[axi], fontsize=20)
        ax.axis('off')

    for i in range(len(output_images[-1])):
        ax = axs.flatten()[i]
        pt = ax.imshow(output_images[-1])
    plt.show()


#%% Print Noise Pic
output_images = []
# generate initial noise
img_data = gen_noise()
output_images.append(img_data.numpy())

display_features(output_images)

#%%
model.predict(output_images[0])

