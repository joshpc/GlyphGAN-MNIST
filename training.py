import torch
from torch.autograd import grad as torch_grad

import time
import imageio
from imageio import imread
from torchvision.utils import make_grid

import numpy as np

from datasets import get_mnist_dataloaders
from visualization import show_images

# Training Functions

def train_critic(D, G, D_solver, batch_size, noise_size, data, gradient_penalty_weight, losses, noise_dimension, data_type):
    """
    Trains the critic (discriminator.) This is a single iteration step.
    """
    # Prepare our data
    generated_data = G(generate_training_noise(batch_size, noise_dimension, data_type))

    # Forward Pass - Calculate probabilities on real and generated data
    d_real = D(data.type(data_type))
    d_generated = D(generated_data)

    # Calculate gradient penalty
    gradient_penalty = calculate_gradient_penalty(D, data.type(data_type), generated_data, batch_size, gradient_penalty_weight, losses, data_type)
    losses['GP'].append(gradient_penalty.data)

    #TODO: Does this go here or at the start?
    D_solver.zero_grad()

    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
    d_loss.backward()
    losses['D'].append(d_loss.data)

    D_solver.step()


def train_generator(D, G, G_solver, batch_size, data, losses, noise_dimension, data_type):
    """
    Trains the generator. This is a single iteration step.
    """
    G_solver.zero_grad()

    # Prepare our data
    generated_data = G(generate_training_noise(batch_size, noise_dimension, data_type))

    # Forward Pass
    d_generated = D(generated_data)

    loss = -d_generated.mean()
    loss.backward()
    losses['G'].append(loss.data)

    G_solver.step()


def train_epoch(D, G, D_solver, G_solver, batch_size, data_loader, gradient_penalty_weight, losses, noise_dimension, data_type):
    steps = 0
    critic_iterations = 5
    for data, labels in data_loader:
        if len(data) % batch_size != 0:
            continue

        steps += 1

        train_critic(D, G, D_solver, batch_size, noise_dimension, data, gradient_penalty_weight, losses, noise_dimension, data_type)

        if steps % critic_iterations == 0:
            train_generator(D, G, G_solver, batch_size, data, losses, noise_dimension, data_type)


def train(D, G, D_solver, G_solver, batch_size, epoch_count, noise_dimension, data_type, generate_gif=False):
    """
    Main training loop for GlyphGAN
    """

    train_loader, _ = get_mnist_dataloaders(batch_size=batch_size)

    if generate_gif:
      fixed_seed = generate_training_noise(128, noise_dimension, data_type)
      sampled_images = G(fixed_seed)
      training_progress_images = []
      show_images(sampled_images.type(data_type))

    losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
    start_time = time.time()
    for epoch in range(epoch_count):
        epoch_start_time = time.time()

        train_epoch(D, G, D_solver, G_solver, batch_size, train_loader, 10, losses, noise_dimension, data_type)
        print("{} --- G: {:4} | D: {:.4} | GP: {:.4} | GNorm: {:.4} --- Total time: {}".format(int(epoch + 1) , losses['G'][-1], losses['D'][-1], losses['GP'][-1], losses['gradient_norm'][-1], (time.time() - epoch_start_time)))

        if generate_gif:
          # Sample our generator using the fixed seed -- we can watch the improvement over time!
          sampled_images = G(fixed_seed)
          show_images(sampled_images)

          # Save the images for a gif!
          image_grid = make_grid(sampled_images.detach().cpu())
          training_progress_images.append(np.transpose(image_grid.numpy(), (1, 2, 0)))

    print('Total training time: #{}'.format(time.time() - start_time))

    if generate_gif:
      imageio.mimsave('./training_{}_epochs.gif'.format(epoch_count), training_progress_images)


# Helper Functions

def generate_training_noise(batch_size, noise_dimension, data_type):
    return torch.randn((batch_size, noise_dimension)).type(data_type)

def calculate_gradient_penalty(D, real_data, generated_data, batch_size, gradient_penalty_weight, losses, data_type):
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).type(data_type)

    interpolated = (alpha * real_data.data + (1 - alpha) * generated_data.data).type(data_type)
    interpolated.requires_grad = True

    # Calculate probability of interpolated examples
    probability_interpolated = D(interpolated)

    #TODO: Clean up?
    gradients = torch_grad(outputs=probability_interpolated,
                           inputs=interpolated,
                           grad_outputs=torch.ones(probability_interpolated.size()).type(data_type),
                           create_graph=True,
                           retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gradient_penalty_weight * ((gradients_norm - 1) ** 2).mean()
