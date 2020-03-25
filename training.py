import torch
from torch.autograd import grad as torch_grad

import time
import imageio
from imageio import imread
from torchvision.utils import make_grid

import numpy as np

from datasets import get_mnist_dataloaders
from visualization import show_images

import torch.nn as nn

# Training Functions

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def train_critic(D, G, D_solver, batch_size, noise_size, data, class_vectors, gradient_penalty_weight, losses, noise_dimension, data_type):
    """
    Trains the critic (discriminator.) This is a single iteration step.
    """
    # Prepare our data
    generated_data = G(generate_training_noise(batch_size, noise_dimension, data_type, class_vectors))

    # Forward Pass - Calculate probabilities on real and generated data
    real_data = data.type(data_type)
    d_real = D(real_data)
    d_generated = D(generated_data)

    # Calculate gradient penalty
    gradient_penalty = calculate_gradient_penalty(D, real_data, generated_data, batch_size, gradient_penalty_weight, losses, data_type)
    losses['GP'].append(gradient_penalty.data)

    # TODO: Does this go here or at the start?
    D_solver.zero_grad()

    # Calculate the Wasserstein Distance. This is the "Weight Mover's distance" between two distributions.
    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
    d_loss.backward()
    losses['D'].append(d_loss.data)

    D_solver.step()


def train_generator(D, G, G_solver, batch_size, data, class_vectors, losses, noise_dimension, data_type):
    """
    Trains the generator. This is a single iteration step.
    """
    G_solver.zero_grad()

    # Prepare our data
    generated_data = G(generate_training_noise(batch_size, noise_dimension, data_type, class_vectors))

    # Forward Pass
    d_generated = D(generated_data)

    loss = -d_generated.mean()
    loss.backward()
    losses['G'].append(loss.data)

    G_solver.step()

def train_epoch(D, G, D_solver, G_solver, character_classes, batch_size, data_loaders, gradient_penalty_weight, losses, noise_dimension, data_type, critic_iterations, all_loader):
    steps = 0

    for character_class in character_classes:
        class_vectors = torch.zeros((batch_size, len(character_classes))).type(data_type)
        class_vectors[:, character_class] = 1

        generator_iterations = 0
        iterations = 0

        for data, labels in data_loaders[character_class]:
            if len(data) % batch_size != 0:
                continue

            steps += 1
            iterations += 1

            train_critic(D, G, D_solver, batch_size, noise_dimension, data, class_vectors, gradient_penalty_weight, losses, noise_dimension, data_type)

            if steps % critic_iterations == 0:
                generator_iterations += 1
                train_generator(D, G, G_solver, batch_size, data, class_vectors, losses, noise_dimension, data_type)

            # print("{} generator iterations over {} iterations".format(generator_iterations, iterations))

def train(D, G, D_solver, G_solver, batch_size, epoch_count, noise_dimension, data_type, critic_iterations, generate_gif=False):
    """
    Main training loop for GlyphGAN

    - Inputs:
    - `D` The Discriminator (Critic)
    - `G` The Generator
    - `D_solver` optimizer for D
    - `G_solver` optimizer for G
    - `epoch_count` the number of epochs to run this training loop. Each epoch will pass through the data once, training the discriminator each time. It will, every `critic_iterations`, also train the generator.
    """

    character_classes = np.arange(10)

    train_loaders, _, all_loader = get_mnist_dataloaders(batch_size=batch_size, character_classes=character_classes)

    if generate_gif:
        class_vectors = torch.zeros((batch_size, len(character_classes))).type(data_type)
        for i in range(batch_size):
            class_vectors[i, i % len(character_classes)] = 1

        fixed_seed = generate_training_noise(batch_size, noise_dimension, data_type, class_vectors)
        sampled_images = G(fixed_seed)
        training_progress_images = []
        show_images(sampled_images.type(data_type))

    losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
    start_time = time.time()

    for epoch in range(epoch_count):
        epoch_start_time = time.time()

        train_epoch(D, G, D_solver, G_solver, character_classes, batch_size, train_loaders, 10, losses, noise_dimension, data_type, critic_iterations, all_loader)
        print("{} --- G: {:4} | D: {:.4} | GP: {:.4} | GNorm: {:.4} --- Total time: {}".format(int(epoch + 1), losses['G'][-1], losses['D'][-1], losses['GP'][-1], losses['gradient_norm'][-1], (time.time() - epoch_start_time)))

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

def generate_training_noise(batch_size, noise_dimension, data_type, class_vectors):
    noise = torch.randn((batch_size, noise_dimension)).type(data_type)
    return torch.cat((class_vectors, noise), dim=1)

def one_hot_encoding(classes, classes_count):
    batch_size = len(classes)
    encoding = torch.zeros(batch_size, classes_count)
    encoding[np.arange(batch_size), classes] = 1

def calculate_gradient_penalty(D, real_data, generated_data, batch_size, gradient_penalty_weight, losses, data_type):
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).type(data_type)

    # 'interpolated' is x-hat
    interpolated = (alpha * real_data.data + (1 - alpha) * generated_data.data).type(data_type)
    interpolated.requires_grad = True

    # Calculate probability of interpolated examples
    probability_interpolated = D(interpolated)

    # TODO: Clean up?
    gradients = torch_grad(outputs=probability_interpolated,
                           inputs=interpolated,
                           grad_outputs=torch.ones(
                               probability_interpolated.size()).type(data_type),
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
