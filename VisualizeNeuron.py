#%% Dataset Preprocessing
import tensorflow as tf
import matplotlib as mpl
import IPython.display as display
import numpy as np
import PIL


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
    return layer_activations[0,w,h,c]
  if unit == 'pixel':
    return tf.math.reduce_mean(layer_activations[0,w,h])
  elif unit == 'channel':
    return tf.math.reduce_mean(layer_activations[:,:,:,c])
  elif unit == 'neuron_softmax':
    return tf.math.softmax(layer_activations[0,w,h])[c]
  elif unit == 'neuron_constraint':
    nonzero_count = 0.0
    for i in layer_activations[0,w,h]:
      if(i > 0.0):
        nonzero_count += 1.0
    return layer_activations[0,w,h,c] - tf.math.reduce_sum(layer_activations[0,w,h]) / nonzero_count
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
def run_visualization_simple(img, model, steps=100, step_size=0.01,w=0, h=0, c=0, unit='layer'):
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
backbone = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

# Maximize the activations of these layers
layer_names = ['block5_conv4']
layers = [backbone.get_layer(name).output for name in layer_names]

# Create the feature extraction model
sub_model = tf.keras.Model(inputs=backbone.input, outputs=layers)



#%%
# target_img = PIL.Image.open('C:\\Users\\jx830\\OneDrive\\??????\\ntust.jpg')
# target_img = tf.keras.applications.vgg16.preprocess_input(np.array(target_img))
noise_image = gen_noise(dim=300, channel=3)
dream_target_img = run_visualization_simple(noise_image, model=sub_model, 
                                  steps=300, step_size=0.01, unit='channel', c=4)

l = sub_model(tf.expand_dims(gen_noise(dim=300, channel=3),axis=0))
tf.keras.utils.array_to_img(dream_target_img)

# %%
