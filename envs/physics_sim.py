"""This file is adapted from the 3-body gravitational physics simulation in
Jaques et al., Physics-as-Inverse-Graphics: Joint Unsupervised Learning of
Objects and Physics from Video (https://arxiv.org/abs/1905.11169).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from skimage.draw import disk
from skimage.transform import resize


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape

    bordered = 0.5*np.ones([nindex, height+2, width+2, intensity])
    for i in range(nindex):
        bordered[i,1:-1,1:-1,:] = array[i]

    array = bordered
    nindex, height, width, intensity = array.shape

    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def compute_wall_collision(pos, vel, radius, img_size):
    if pos[1] - radius <= 0:
        vel[1] = -vel[1]
        pos[1] = -(pos[1] - radius) + radius
    if pos[1] + radius >= img_size[1]:
        vel[1] = -vel[1]
        pos[1] = img_size[1] - (pos[1] + radius - img_size[1]) - radius
    if pos[0] - radius <= 0:
        vel[0] = -vel[0]
        pos[0] = -(pos[0] - radius) + radius
    if pos[0] + radius >= img_size[0]:
        vel[0] = -vel[0]
        pos[0] = img_size[0] - (pos[0] + radius - img_size[0]) - radius
    return pos, vel


def verify_wall_collision(pos, vel, radius, img_size):
    if pos[1] - radius <= 0:
        return True
    if pos[1] + radius >= img_size[1]:
        return True
    if pos[0] - radius <= 0:
        return True
    if pos[0] + radius >= img_size[0]:
        return True
    return False


def verify_object_collision(poss, radius):
    for pos1, pos2 in combinations(poss, 2):
        if np.linalg.norm(pos1 - pos2) <= radius:
            return True
    return False


def create_frame(poss, cifar_background, img_size, radius, cifar_img=None, color=False):
    scale = 10
    scaled_img_size = [img_size[0] * scale, img_size[1] * scale]
    
    if cifar_background:
        frame = cifar_img
        frame = rgb2gray(frame) / 255
        frame = resize(frame, scaled_img_size)
        frame = np.clip(frame - 0.2, 0.0, 1.0)  # darken image a bit
    else:
        if color:
            frame = np.zeros(scaled_img_size + [3], dtype=np.float32)
        else:
            frame = np.zeros(scaled_img_size + [1], dtype=np.float32)

    for j, pos in enumerate(poss):
        rr, cc = disk((int(pos[1] * scale), int(pos[0] * scale)), 
                     radius * scale, shape=scaled_img_size)
        if color:
            frame[rr, cc, 2 - j] = 1.0
        else:
            frame[rr, cc, 0] = 1.0

    frame = resize(frame, img_size, anti_aliasing=True)
    frame = (frame * 255).astype(np.uint8)
    return frame


def generate_nbody_problem_dataset(dest,
                                 n_bodies,
                                 train_set_size,
                                 valid_set_size,
                                 test_set_size,
                                 seq_len,
                                 img_size=None,
                                 radius=3,
                                 dt=0.3,
                                 display_dt=2.0,
                                 g=9.8,
                                 masses=None,
                                 vx0_max=0.0,
                                 vy0_max=0.0,
                                 color=False,
                                 cifar_background=False,
                                 ode_steps=10,
                                 seed=0):
    """
    Generates n-body gravitational problem dataset.
    Supports both 2-body and 3-body problems.
    """
    np.random.seed(seed)

    if cifar_background:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if img_size is None:
        img_size = [32, 32]

    if masses is None:
        masses = [1.0] * n_bodies
    assert len(masses) == n_bodies, f"Must provide exactly {n_bodies} masses"

    def generate_sequence():
        collision = True
        while collision:
            seq = []
            accumulated_time = 0.0
            next_frame_time = 0.0

            # Initialize positions
            cm_pos = np.array(img_size) / 2
            base_angle = np.random.rand() * 2 * np.pi
            angles = [base_angle + i * (2 * np.pi / n_bodies) + 
                     (np.random.rand() - 0.5) / 2 for i in range(n_bodies)]

            # Calculate positions
            r = (np.random.rand() / 2 + 0.75) * img_size[0] / 4
            poss = np.array([[np.cos(angle) * r + cm_pos[0], 
                             np.sin(angle) * r + cm_pos[1]] for angle in angles])

            # Initialize velocities
            r = np.random.randint(0, 2) * 2 - 1
            angles = [angle + r * np.pi / 2 for angle in angles]
            noise = np.random.rand(2) - 0.5
            vels = np.array([[np.cos(angle) * vx0_max + noise[0],
                             np.sin(angle) * vy0_max + noise[1]] for angle in angles])

            if cifar_background:
                cifar_img = x_train[np.random.randint(50000)]

            # Save initial frame
            frame = create_frame(poss, cifar_background, img_size, radius, 
                               cifar_img if cifar_background else None, color)
            seq.append(frame)
            next_frame_time += display_dt

            # Simulate for seq_len-1 more frames
            while len(seq) < seq_len:
                # Rollout physics
                for _ in range(ode_steps):
                    # Calculate forces between all pairs
                    F = np.zeros((n_bodies, 2))
                    for i in range(n_bodies):
                        for j in range(n_bodies):
                            if i != j:
                                r = poss[i] - poss[j]
                                norm = np.linalg.norm(r)
                                F[i] += -g * masses[j] * r / norm**3

                    # Update velocities and positions
                    vels = vels + dt/ode_steps * (F / np.array(masses)[:, np.newaxis])
                    poss = poss + dt/ode_steps * vels

                    collision = any([verify_wall_collision(pos, vel, radius, img_size) 
                                   for pos, vel in zip(poss, vels)]) or \
                              verify_object_collision(poss, radius + 1)
                    
                    if collision:
                        break

                if collision:
                    break

                accumulated_time += dt
                
                # Check if we should save a frame
                if accumulated_time >= next_frame_time:
                    frame = create_frame(poss, cifar_background, img_size, radius,
                                       cifar_img if cifar_background else None, color)
                    seq.append(frame)
                    next_frame_time += display_dt

            if not collision and len(seq) == seq_len:
                break

        return np.array(seq)

    # Generate datasets
    datasets = {
        'train': [generate_sequence() for _ in range(train_set_size)],
        'valid': [generate_sequence() for _ in range(valid_set_size)],
        'test': [generate_sequence() for _ in range(test_set_size)]
    }

    # Save datasets
    if dest.endswith('.npz'):
        np.savez_compressed(dest,
                          train_x=np.array(datasets['train']),
                          valid_x=np.array(datasets['valid']),
                          test_x=np.array(datasets['test']))
    else:
        for split, data in datasets.items():
            if len(data) > 0:
                np.save(f"{dest}_{split}.npy", np.array(data))

    # Save sample visualization
    if train_set_size > 0:
        # Create figure to show 10 sequences with 10 frames each
        fig = plt.figure(figsize=(20, 20))
        for i in range(10):  # 10 sequences
            if i < len(datasets['train']):
                sequence = datasets['train'][i]
                for j in range(10):  # 10 frames per sequence
                    if j < len(sequence):
                        ax = plt.subplot(10, 10, i*10 + j + 1)
                        ax.imshow(sequence[j])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(dest.split(".")[0] + "_samples.jpg")
        plt.close(fig)