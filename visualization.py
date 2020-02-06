import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec

plot.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plot.rcParams['image.interpolation'] = 'nearest'
plot.rcParams['image.cmap'] = 'gray'

import numpy as np

def show_images(generator_output, image_count=16):
  images_data = generator_output.data.cpu().numpy()[0:image_count]

  images = np.reshape(images_data, [images_data.shape[0], -1])  # images reshape to (batch_size, D)
  sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
  sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

  grid_spec = gridspec.GridSpec(sqrtn, sqrtn)
  grid_spec.update(wspace=0.05, hspace=0.05)

  for i, img in enumerate(images):
      ax = plot.subplot(grid_spec[i])
      plot.axis('off')
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_aspect('equal')
      plot.imshow(img.reshape([sqrtimg,sqrtimg]))

  plot.show()
  print()