#%% Dataset Preprocessing
import tensorflow as tf
import matplotlib as mpl
import IPython.display as display
import numpy as np


#%%
# Display an image
def gen_noise(dim=28, channel = 1):
    input_img_data = tf.random.uniform((dim, dim, channel))
    return tf.cast(input_img_data, tf.float32).numpy()

#%%
def calc_loss(img, model, w=0, h=0, c=0, unit='layer'):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  
  if unit == 'neuron':
    return tf.math.reduce_mean(layer_activations[0,w,h,:])
  elif unit == 'channel':
    return tf.math.reduce_mean(layer_activations[:,:,:,c])
  else: #layer
    return tf.math.reduce_mean(layer_activations)

#%%
class FeatureVisualization(tf.Module):
  def __init__(self, model, w=0, h=0, c=0, unit='layer'):
    self.model = model
    self.w = w
    self.h = h
    self.c = c
    self.unit = unit

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
          loss = calc_loss(img, self.model, self.w, self.h, self.c, self.unit)

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
def run_deep_dream_simple(img, model, steps=100, step_size=0.01,w=0, h=0, c=0, unit='layer'):
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

    featureVisualization = FeatureVisualization(model, w, h, c, unit)
    loss, img = featureVisualization(img, run_steps, tf.constant(step_size))

    # display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))
    tf.keras.utils.array_to_img(img)
  return img

#%%
backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

# Maximize the activations of these layers
layer_names = ['block2_conv1']
layers = [backbone.get_layer(name).output for name in layer_names]

# Create the feature extraction model
sub_model = tf.keras.Model(inputs=backbone.input, outputs=layers)



#%%
import PIL
dog_img = PIL.Image.open('C:\\Users\\jx830\\OneDrive\\桌面\\YellowLabradorLooking_new.jpg')
tf.keras.utils.array_to_img(dog_img)
dog_img = tf.keras.applications.vgg16.preprocess_input(np.array(dog_img))

dream_inception_dog_img = run_deep_dream_simple(gen_noise(dim=300,channel=3), model=sub_model, 
                                  steps=300, step_size=0.01)

l = sub_model(tf.expand_dims(dog_img,axis=0))



tf.keras.utils.array_to_img(dream_inception_dog_img)

# %%
