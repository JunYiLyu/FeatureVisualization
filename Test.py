#%% Dataset Preprocessing
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

tf.config.run_functions_eagerly(True)


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


#%% Build Model
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


#%% 產生雜訊

def gen_noise(dim=28, nex = 1):
    input_img_data = tf.random.uniform((nex, dim, dim, 1))
    return tf.Variable(tf.cast(input_img_data, tf.float32))

#%% 畫圖
import matplotlib.pyplot as plt
import numpy as np

def display_features(output_images, filter_titles=None, ncols=10, zoom = 5, sat_exp=2.0, val_exp = 1.0):
    nrows = int(np.ceil(len(output_images[0]) / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5,nrows*5))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace = 0.1, wspace = 0.01)
    for axi, ax in enumerate(axs.flatten()):
        if filter_titles is not None:
            if axi < len(filter_titles):
                ax.set_title(filter_titles[axi], fontsize=20)
        ax.axis('off')

    for i in range(len(output_images[0])):
        ax = axs.flatten()[i]
        pt = ax.imshow(output_images[0][i])
    plt.show()


#%% Print Noise Pic
output_images = []
img_data = gen_noise()
output_images.append(img_data.numpy())
display_features(output_images)

#%% print ds_train
data = list(ds_train)[0] #call 一次 ds_train 就會shuffle ?
display_features(data)
model.predict(data[0])[1]


#%% def upscale opt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian

def upscale_image(imgs, upscaling_factor=1.1, sigma=0):
    """ Upsample and smooth the list of images
    """
    img_list = []
    for img in imgs:
        if upscaling_factor == 1.0:
            upscaled_img = img
        else:
            sz = np.array(np.shape(img))[1]
            sz_up = (upscaling_factor * sz).astype("int")
            lower = int(np.floor((sz_up - sz) / 2))
            upper = int(np.ceil((sz_up - sz) / 2))

            upscaled_img = resize(img.astype("float"), (sz_up, sz_up), anti_aliasing=True)
            upscaled_img = upscaled_img[
                lower:-lower, lower:-lower, :,
            ]
        if sigma > 0:
            upscaled_img = gaussian(upscaled_img, sigma=sigma, multichannel=True)
        img_list.append(upscaled_img)
    return tf.Variable(tf.cast(img_list, tf.float32))

def zero_one_norm(x):
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
def z_score(x, scale = 1.0):
    return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x)/scale)
def norm(x):
    return zero_one_norm(z_score(x))
def soft_norm(x, n_std = 15):
    """ zscore and set n_std range of 0-1, then clip
    """
    x = z_score(x) / (n_std*2)
    return tf.clip_by_value(x + 0.5, 0, 1)


def get_opt_function():
    """ This function returns the optimizer function. This is just necessary because of tensorflow weirdness. 
    See: https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
    """
    @tf.function
    def opt(
        submodel,
        input_data,
        filter_index,
        optimizer,
        steps=100,
        lr=0.01,
        layer_dims=2,
        soft_norm_std=15,
        norm_f = "soft_norm",
        normalize_grads = True
    ):
        """ This function runs a single optimization over the list of images
        """
        loss_value = 0

        # determine if this is a convolutional, or fully connected layer
        if layer_dims == 2:
            # identity function because second dimension is already the filter dimension
            loss_func = lambda out_: out_
        if layer_dims == 4:
            # flip to make filter dimension second dimension
            loss_func = lambda out_: tf.transpose(out_, perm = [0,3,1,2])
        # optimization
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(input_data)
                
                if norm_f == "sigmoid":
                    outputs = submodel(tf.nn.sigmoid(input_data))
                else:
                    outputs = submodel(input_data)
                outputs = loss_func(outputs)

                loss_value = tf.gather_nd(
                    outputs, indices=filter_index, batch_dims=0, name=None
                )
                grads = tape.gradient(loss_value, input_data)
                
                if normalize_grads:
                    norm_divisor = tf.expand_dims(
                        tf.expand_dims(
                            tf.expand_dims(tf.math.reduce_std(grads, axis=[1, 2, 3]), 1), 1
                        ),
                        1,
                    )

                    normalized_grads = grads / norm_divisor
                    optimizer.apply_gradients(zip([-normalized_grads], [input_data]))
                else: 
                    optimizer.apply_gradients(zip([-grads], [input_data]))
  
                if norm_f == "clip":
                    input_data.assign(tf.clip_by_value(input_data, 0, 1))
                elif norm_f == "soft_norm":
                    input_data.assign(soft_norm(input_data, n_std=soft_norm_std))
                elif norm_f == "sigmoid":
                    input_data.assign(tf.clip_by_value(input_data, -50, 50))
                    # normalizes color channels to sit scale roughly uniform
                    # input_data.assign(z_score(input_data, scale=1.5))
                  
 
                    
        return input_data, loss_value
        
    return opt



#%% 
from tqdm.autonotebook import tqdm

def optimize_filter(
    submodel,
    layer_name,
    filter_index,
    filters_shape,
    steps=20,
    lr=0.01,
    layer_dims=2,
    n_upsample=1,
    sigma=0,
    upscaling_factor=1.1,
    soft_norm_std=15,
    single_receptive_field=True,
    norm_f = "soft_norm",
    normalize_grads = True
):
    """ This pulls together the steps for optimizing the image, and upsampling
    """

    opt = get_opt_function()
    optimizer = tf.keras.optimizers.Adam(lr)

    # subset center neuron if we only want to look at one receptive field
    if single_receptive_field & (layer_dims == 4):
        filter_index = [
            [i[0], i[1], int(filters_shape[1] / 2), int(filters_shape[2] / 2)]
            for i in filter_index
        ]
        
    loss_list = []
    # list of outputs during optimization
    output_images = []
    # generate initial noise
    img_data = gen_noise(nex=len(filter_index))
    output_images.append(img_data.numpy())
    
    # apply optimization
    for i in tqdm(range(n_upsample), leave=False):
        
        # optimize
        img_data, loss = opt(
            submodel,
            img_data,
            filter_index,
            optimizer=optimizer,
            steps=steps,
            lr=lr,
            layer_dims=layer_dims,
            soft_norm_std=soft_norm_std,
            norm_f = norm_f,
            normalize_grads=normalize_grads
        )  

        loss_list.append(np.mean(loss))
        output_images.append(img_data.numpy())
        
        # upsample
        if i < (n_upsample - 1):
            img_data = upscale_image(
                img_data.numpy(), upscaling_factor=upscaling_factor, sigma=sigma,
            )


        # if tf.is_tensor(img_data):
        #    img_data = tf.Variable(tf.cast(img_data.numpy(), tf.float32))
    
    # # brg to rgb color channel conversion
    # if norm_f == "sigmoid":
    #     output_images = [tf.nn.sigmoid(i).numpy()[:,:,:,::-1] for i in output_images]
    # else:
    #     output_images = [i[:,:,:,::-1] for i in output_images]

    return output_images, loss_list

#%%
layer_name = "dense_1"

filter_index = [[0,0]]

# get module of input/output
submodel = tf.keras.models.Model(
    [model.inputs[0]], [model.get_layer(layer_name).output]
)

filters_shape = submodel.outputs[0].shape
output_images, loss_list = optimize_filter(
    submodel,
    layer_name,
    filter_index,
    filters_shape=filters_shape,
    steps = 20, # how many training steps to perform
    lr=0.1, # gradient step size 
    layer_dims=len(submodel.outputs[0].shape), # how many dimensions the output layer is (2 for fully connected, 4 for convolutional)
    n_upsample=50, # how many steps to upsample
    sigma=1.0, # the amount of blurring to perform when upsampling
    upscaling_factor=1.1, # how much to upsample by
    single_receptive_field=False, # whether to optimize a single neuron, or optimize over the layer
    norm_f = "sigmoid", # how to normalize/color channels between 0 and 1 (clip, sigmoid, )
    soft_norm_std = 3, # the number of standard deviations to clip if norm_f is soft_norm (lower = more saturated)
    normalize_grads=True
    
)


loss_list

#%%

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
        # rgb = adjust_hsv(output_images[-1][i], sat_exp = sat_exp, val_exp = val_exp)
        pt = ax.imshow(output_images[-1][i])
    plt.show()

display_features(output_images, 'DENSE', ncols=4, zoom = 5)