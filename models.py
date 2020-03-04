import torch
import torch.nn as nn
import torch.optim as optim

from model_util import Unflatten, Flatten

def build_glyph_gan_generator(image_size=(32, 32, 1), noise_dimension=100, character_class_count=10, dimension=16):
  """
  PyTorch model implementing the GlyphGAN generator.

  It will generate images of size `image_size`.

  Inputs:
  - `image_size`: A tuple (W,H,C) for the size of the images. Defaults to (32, 32, 1)
  - `noise_dimension`: The width of the noise. Defaults to 100.
  - `dimension`: The depth of the noise. Defaults to 16.
  """
  feature_sizes = (image_size[0] / 16, image_size[1] / 16)
  output_size = int(8 * dimension * feature_sizes[0] * feature_sizes[1])

  return nn.Sequential(
    nn.Linear(noise_dimension + character_class_count, output_size),
    Unflatten(C=int(8 * dimension), W=int(feature_sizes[0]), H=int(feature_sizes[1])),
    nn.BatchNorm2d(8 * dimension),

    # Fractionally Strided Conv 1
    nn.ConvTranspose2d(8 * dimension, 4 * dimension, 4, 2, 1),
    nn.BatchNorm2d(4 * dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 2
    nn.ConvTranspose2d(4 * dimension, 2 * dimension, 4, 2, 1),
    nn.BatchNorm2d(2 * dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 3
    nn.ConvTranspose2d(2 * dimension, dimension, 4, 2, 1),
    nn.BatchNorm2d(dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 4
    nn.ConvTranspose2d(dimension, image_size[2], 4, 2, 1),
    nn.Sigmoid()
  )

def build_glyph_gan_critic(image_size=(32, 32), dimension=16):
  """
  PyTorch model implementing the GlyphGAN critic.

  Inputs:
  - `image_size`: The size of the images (W, H) tuple. (32, 32)
  - `dimension`: The dpeth of the noise, defaults to 16.
  """

  output_size = int(8 * dimension * (image_size[0] / 16) * (image_size[1] / 16))

  return nn.Sequential(
    nn.Conv2d(image_size[2], dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(dimension, 2 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * dimension, 4 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * dimension, 8 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    Flatten(),

    nn.Linear(output_size, 1),
    nn.Sigmoid()
  )

def get_optimizer(model, learning_rate=2e-4, beta1=0.5, beta2=0.99):
    """
    Adam optimizer for model

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    return optimizer
