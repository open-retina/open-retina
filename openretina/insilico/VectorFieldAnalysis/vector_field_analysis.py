from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from PIL import Image
from scipy.ndimage import zoom
import torch

from openretina.models.core_readout import load_core_readout_from_remote


def prepare_movies_dataset(
    model, session_id, n_image_frames=16, normalize_movies=True, image_library=None, device="cuda"
):
    n_channels = model.data_info["input_shape"][0]

    target_h, target_w = model.data_info["input_shape"][1:3]

    print(f"Model input shape: {model.data_info['input_shape']}")
    if image_library is None:
        # Load the model if not provided
        image_dir = (
            "/home/baptiste/Documents/LabPipelines/open-retina/openretina/insilico/VectorFieldAnalysis/natural_images/"
        )
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])

        images = np.array([np.array(Image.open(os.path.join(image_dir, f))) for f in image_files])
        print(f"Loaded {len(images)} images with shape: {images.shape}")
        print(
            f"Image stats - Mean: {images.mean():.2f}, Std: {images.std():.2f}, Min: {images.min()}, Max: {images.max()}"
        )
        compressed_images = []
        for img in images:
            # Downsample and center crop
            downsample_factor = int(min(img.shape[0] / target_h, img.shape[1] / target_w))
            img = img[::downsample_factor, ::downsample_factor]

            h, w = img.shape
            start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
            compressed_images.append(img[start_h : start_h + target_h, start_w : start_w + target_w])

        compressed_images = np.array(compressed_images)
        compressed_images = np.repeat(compressed_images[:, np.newaxis, :, :], n_channels, axis=1).astype(np.float32)
        print(f"Compressed shape: {compressed_images.shape}")

    else:
        # Use the provided image library
        compressed_images = image_library
        print(f"Using provided image library with shape: {compressed_images.shape}")
        print(
            f"Image stats- Mean: {compressed_images.mean():.2f}, Std: {compressed_images.std():.2f}, Min: {compressed_images.min()}, Max: {compressed_images.max()}"
        )

    # Determine temporal padding needed by model
    empty_movie = torch.zeros(1, n_channels, 100, target_h, target_w, dtype=torch.float32, device=device)
    n_empty_frames = 100 - model(empty_movie).shape[1] +10 # +10 for border effects
    print(f"Number of empty frames needed: {n_empty_frames}")

    movies = np.repeat(compressed_images[:, :, np.newaxis, :, :], n_empty_frames + n_image_frames, axis=2)

    # Normalize using model parameters
    if normalize_movies:
        if n_channels == 1:
            stim_mean = model.data_info["movie_norm_dict"][session_id]["norm_mean"]
            stim_std = model.data_info["movie_norm_dict"][session_id]["norm_std"]
        else:
            print("unclear behavior when n_channels > 1")
            stim_mean = model.data_info["movie_norm_dict"]["default"]["norm_mean"]
            stim_std = model.data_info["movie_norm_dict"]["default"]["norm_std"]

        for channel in range(n_channels):
            movies[:, channel, :, :, :] = (movies[:, channel, :, :, :] - stim_mean) / stim_std

    # Set initial frames to mean grey in the images
    movies[:, :, :n_empty_frames, :, :] = movies.mean()
    print(f"Final movies shape: {movies.shape}")

    return movies, n_empty_frames


def compute_lsta_library(model, movies, session_id, cell_id, batch_size=64, integration_window=(5, 10), device="cuda"):
    model.eval()
    batch_size = 64
    all_lstas = []
    all_outputs = []

    # movies = movies[:1000]  # For debugging, limit to 1000 movies, TO DO: Suppress this line

    for i in range(0, len(movies), batch_size):
        batch_movies = torch.tensor(movies[i : i + batch_size], dtype=torch.float32, device=device)
        batch_movies.requires_grad = True

        outputs = model(batch_movies, data_key=session_id)
        chosen_cell_outputs = outputs[
            :, integration_window[0]:integration_window[1], cell_id
        ].sum()  # I usually give up the first frame since it is contaminated by past responses (not done here)
        chosen_cell_outputs.backward()

        batch_lstas = batch_movies.grad.detach().cpu().numpy()
        all_lstas.append(batch_lstas)
        all_outputs.append(outputs.detach().cpu().numpy())

        # Clear gradients
        batch_movies.grad.zero_()

    ex_lsta = np.concatenate(all_lstas, axis=0)
    lsta_library = ex_lsta.mean(axis=2)
    response_library = np.concatenate(all_outputs, axis=0)
    return lsta_library, response_library


def get_pc_from_pca(model, channel, lsta_library, plot=False):
    # Select channel and reshape
    lsta_reshaped = lsta_library[:, channel, :, :].reshape(lsta_library.shape[0], -1)

    pca = PCA(n_components=2)
    pca.fit(lsta_reshaped)

    explained_variance = pca.explained_variance_ratio_
    PC1, PC2 = pca.components_

    if plot:
        PC_max = max(np.abs(PC1).max(), np.abs(PC2).max())
        plt.figure(figsize=(10, 5))
        for k in range(2):
            plt.subplot(1, 2, k + 1)
            plt.imshow(
                pca.components_[k].reshape(model.data_info["input_shape"][1:3]), cmap="bwr", vmin=-PC_max, vmax=PC_max
            )
            plt.title(f"PCA {k} ({explained_variance[k]:.2f} e.v.)")
            plt.axis("off")

    return PC1, PC2, explained_variance


def get_images_coordinate(images, PC1, PC2, plot=False):
    resh_training = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    images_coordinate = []
    for image in resh_training:
        images_coordinate.append([np.dot(PC1, image), np.dot(PC2, image)])
    images_coordinate = np.array(images_coordinate)

    if plot:
        pt_x = images_coordinate[:, 0]
        pt_y = images_coordinate[:, 1]
        plt.figure()
        plt.scatter(pt_x, pt_y)

    return images_coordinate


def plot_untreated_vectorfield(lsta_library, PC1, PC2, images):
    arrowheads = np.array([[np.dot(PC1, lsta.flatten()), np.dot(PC2, lsta.flatten())] for lsta in lsta_library])
    plt.figure(figsize=(20, 15))
    arrowheads = arrowheads * 1000
    plt.quiver(
        images[: len(lsta_library), 0],
        images[: len(lsta_library), 1],
        arrowheads[:, 0],
        arrowheads[:, 1],
        width=0.002,
        scale_units="xy",
        angles="xy",
        scale=arrowheads.max() / 1.5,
        alpha=0.2,
    )
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.axis("off")
    return plt.gcf()


def plot_clean_vectorfield(
    lsta_library,
    channel,
    PC1,
    PC2,
    images,
    images_coordinate,
    explained_variance,
    x_bins=31,
    y_bins=31,
):
    dx = 40 / x_bins
    dy = 40 / x_bins

    binned_imgs = []
    binned_lstas = []
    x_size = lsta_library.shape[-2]
    y_size = lsta_library.shape[-1]

    for x_tick in range(x_bins):
        x_val = -20 + x_tick * dx
        for y_tick in range(y_bins):
            y_val = -20 + y_tick * dy
            temp_img = np.zeros((x_size, y_size))
            temp_lsta = np.zeros((x_size, y_size))
            nb = 0
            for i, coords in enumerate(images_coordinate[: lsta_library.shape[0]]):
                if x_val <= coords[0] < x_val + dx and y_val <= coords[1] < y_val + dy:
                    temp_img += images[i]
                    temp_lsta += lsta_library[i, channel]
                    nb += 1
            if nb > 0:
                binned_imgs.append(temp_img / nb)
                binned_lstas.append(temp_lsta / nb)

    binned_imgs = np.array(binned_imgs)
    binned_lstas = np.array(binned_lstas)
    # Check if we have any binned data
    if len(binned_imgs) == 0:
        print("Warning: No images found in coordinate bins. Try adjusting bin size or coordinate range.")
        return None

    resh_binned_imgs = binned_imgs.reshape(binned_imgs.shape[0], -1)
    resh_binned_lstas = binned_lstas.reshape(binned_lstas.shape[0], -1)

    binned_arrowtails = np.array([[np.dot(PC1, img), np.dot(PC2, img)] for img in resh_binned_imgs])
    binned_arrowheads = np.array([[np.dot(PC1, lsta), np.dot(PC2, lsta)] for lsta in resh_binned_lstas])

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.quiver(
        binned_arrowtails[:, 0],
        binned_arrowtails[:, 1],
        binned_arrowheads[:, 0],
        binned_arrowheads[:, 1],
        color="black",
        width=0.002,
        scale_units="xy",
        angles="xy",
        scale=binned_arrowheads.max(),
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add arrowheads to axes using matplotlib arrow function
    xlim = max(np.abs(binned_arrowtails).max(), np.abs(images_coordinate).max()) * 1.1
    ax.arrow(
        -xlim * 0.75, 0, 1.5 * xlim, 0, head_width=xlim * 0.02, head_length=xlim * 0.02, fc="k", ec="k", linewidth=1
    )
    ax.arrow(
        0, -xlim * 0.75, 0, 1.5 * xlim, head_width=xlim * 0.02, head_length=xlim * 0.02, fc="k", ec="k", linewidth=1
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-xlim, xlim])

    # Add PC components as insets
    PC_max = max(np.abs(PC1).max(), np.abs(PC2).max())

    ax_img1 = fig.add_axes([0.825, 0.425, 0.15, 0.15], anchor="C", zorder=1)
    ax_img1.imshow(PC1.reshape(x_size, y_size), cmap="bwr", vmin=-PC_max, vmax=PC_max)
    ax_img1.axis("off")
    ax_img1.set_title(f"PC1 ({explained_variance[0]:.1%})", size=20)

    ax_img2 = fig.add_axes([0.425, 0.825, 0.15, 0.15], anchor="C", zorder=1)
    ax_img2.imshow(PC2.reshape(x_size, y_size), cmap="bwr", vmin=-PC_max, vmax=PC_max)
    ax_img2.axis("off")
    ax_img2.set_title(f"PC2 ({explained_variance[1]:.1%})", size=20)
    return fig
