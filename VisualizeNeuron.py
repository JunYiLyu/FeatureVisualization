#%% Dataset Preprocessing
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import matplotlib as mpl
import IPython.display as display
import numpy as np

vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

#%%
# Display an image
def gen_noise(dim=28, channel = 1):
    input_img_data = tf.random.uniform((dim, dim, channel))
    return tf.cast(input_img_data, tf.float32).numpy()
    

tf.keras.utils.array_to_img(
    gen_noise(dim=225, channel=3)
)

#%%
# Maximize the activations of these layers
inception_names = ['block2_conv1']
inception_layers = [vgg.get_layer(name).output for name in inception_names]

# Create the feature extraction model
dream_vgg_model = tf.keras.Model(inputs=vgg.input, outputs=inception_layers)


#%%
def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  
  # pixel tf.math.reduce_mean(layer_activations[0,0,0,:])
  # chanel tf.math.reduce_mean(layer_activations[:,:,:,2])

  return  tf.math.reduce_mean(layer_activations[0,0,0,:])

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img

#%%
def run_deep_dream_simple(img, model, steps=100, step_size=0.01):
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    deepdream = DeepDream(model)
    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    # display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))
    tf.keras.utils.array_to_img(img)
  return img



#%%
import PIL
dog_img = PIL.Image.open('C:\\Users\\jx830\\OneDrive\\桌面\\YellowLabradorLooking_new.jpg')
tf.keras.utils.array_to_img(dog_img)
dog_img = tf.keras.applications.vgg16.preprocess_input(np.array(dog_img))

dream_inception_dog_img = run_deep_dream_simple(gen_noise(dim=300,channel=3), model=dream_vgg_model, 
                                  steps=600, step_size=0.01)

l = dream_vgg_model(tf.expand_dims(dog_img,axis=0))



tf.keras.utils.array_to_img(dream_inception_dog_img)

# %%
