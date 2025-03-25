import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.fft import fft2, ifft2, fftshift
import skimage
import skimage.io
import itertools
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
import natsort
import pprint

def pad_image(image, pad_shape, center=False):
    """Pad an image with zeros to match the desired shape."""
    pad_size = (pad_shape[0] - image.shape[0], pad_shape[1] - image.shape[1])
    is_rgb = image.ndim > 2
    if center:
        left_pad = int(pad_size[1] // 2)
        right_pad = int(pad_size[1] - left_pad)
        top_pad = int(pad_size[0] // 2)
        bottom_pad = int(pad_size[0] - top_pad)
    else:
        left_pad = 0
        right_pad = pad_size[1]
        top_pad = 0
        bottom_pad = pad_size[0]
    if is_rgb:
        padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
    return padded_image, (left_pad, right_pad, top_pad, bottom_pad)


def combine(composite, f2, offset):
    """Combine two fragments with an offset."""
    # Offset is number of pixels to shift f2 to align with f1
    # Offset is the absolute offset of f2 relative to composite
    # Calculate the expected combined image size using the offset
    f2_shape = f2.shape
    
    start_y = offset[0]
    start_x = offset[1]
    end_y = start_y + f2_shape[0]
    end_x = start_x + f2_shape[1]
    top_pad = int(-min(0, start_y))
    left_pad = int(-min(0, start_x))
    bottom_pad = int(max(0, end_y - composite.shape[0]))
    right_pad = int(max(0, end_x - composite.shape[1]))
    
    new_composite = np.pad(composite, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
    counts = np.zeros(new_composite.shape)
    counts[top_pad:top_pad+composite.shape[0], left_pad:left_pad+composite.shape[1]] = 1
    # Place new fragment in the new composite
    start_r = int(top_pad + start_y)
    start_c = int(left_pad + start_x)
    end_r = int(start_r + f2_shape[0])
    end_c = int(start_c + f2_shape[1])
    new_composite[start_r:end_r, start_c:end_c] += f2
    counts[start_r:end_r, start_c:end_c] += 1
    counts[counts == 0] = 1
    new_composite /= counts
    return new_composite


phase_correlation_cache = {}
def phase_correlation(image_a, image_b, correlation_threshold=0., downsample_factor=1, use_skimage=False, full_convolution=True):
    """
    Compute the relative translation between image_a and image_b
    using phase correlation. Both images must have the same shape.

    Returns:
        shift: np.array([dy, dx]) estimated shift such that
               image_b = shift(image_a). The returned shift reflects
               the translation to apply to image_b relative to image_a.

    Reference:
       Kuglin, C. D., & Hines, D. C. (1975). The phase correlation technique.
       [See also: https://docs.opencv.org/]
    """
    # key = (image_a.tobytes(), image_b.tobytes())
    # if key in phase_correlation_cache:
    #     return phase_correlation_cache[key]

    # Compute FFTs of both images.
    pad_shape = np.array([max(image_a.shape[0], image_b.shape[0]), max(image_a.shape[1], image_b.shape[1])])
    
    image_a = rgba2gray(image_a)
    image_b = rgba2gray(image_b)
    
    if use_skimage:
        # Pad images to the same size, centered on the middle of the output shape
        image_a, _ = pad_image(image_a, pad_shape)
        image_b, _ = pad_image(image_b, pad_shape)
        shift, error, phasediff = phase_cross_correlation(image_a, image_b, disambiguate=True, overlap_ratio=correlation_threshold, normalization=None)
        shift = shift * downsample_factor
        return shift, -error #-error - phasediff
    
    fft_shape = 2 * pad_shape + 1 if full_convolution else pad_shape
    
    F_a = fft2(image_a, fft_shape, norm='ortho')
    F_b = fft2(image_b, fft_shape, norm='ortho')
    # Compute the cross-power spectrum
    cross_power = F_a * np.conj(F_b)
    # Normalize to get only phase information (add a small constant to prevent division by zero)
    cross_power /= (np.abs(cross_power) + 1e-8)
    # Inverse FFT to obtain correlation
    corr = ifft2(cross_power)
    # For ease of peak finding, shift the zero-frequency component to the center
    corr = fftshift(corr)

    # Find peak position in correlation
    max_corr = np.max(np.abs(corr))
    if max_corr < correlation_threshold:
        return np.array([0, 0]), 0
    max_pos = np.unravel_index(np.argmax(np.abs(corr).flatten()), corr.shape)
    midpoints = np.array(corr.shape) // 2
    shift = np.array(max_pos) - midpoints
    shift = shift * downsample_factor
    return shift, max_corr


def rgba2gray(image):
    im = skimage.color.rgba2rgb(image)
    return skimage.color.rgb2gray(im)


def stitch_images(fragments, downsample_factor=1, order=1, use_skimage=False):
    """Stitch a list of fragments together."""
    # Crop the fragments to be divisible by downsample_factor
    fragments = [fragment[:(fragment.shape[0] // downsample_factor) * downsample_factor, :(fragment.shape[1] // downsample_factor) * downsample_factor] for fragment in fragments]

    # Downsample the fragments
    downsampled_fragments = Parallel(n_jobs=-1)(delayed(skimage.transform.resize)(fragment, (fragment.shape[0] // downsample_factor, fragment.shape[1] // downsample_factor), order=order, anti_aliasing=True if order > 0 else False) \
        for fragment in tqdm(fragments, desc="Downsampling fragments"))
    assert len(fragments) == len(downsampled_fragments)
    # Iterate through the fragments and compute the offset for each fragment against each other fragment
    error_matrix = 1e6 * np.ones((len(fragments), len(fragments)))
    offset_matrix = np.zeros((len(fragments), len(fragments), 2))
    # Do in parallel with joblib    
    f1_f2_idx_combinations = list(itertools.combinations(range(len(fragments)), 2))
    offset_max_corr_results = Parallel(n_jobs=-1)(delayed(phase_correlation)(downsampled_fragments[f1_idx], downsampled_fragments[f2_idx], downsample_factor=downsample_factor, use_skimage=use_skimage) for f1_idx, f2_idx in tqdm(f1_f2_idx_combinations))
    # Unpack the results
    for (f1_idx, f2_idx), (offset, max_corr) in zip(f1_f2_idx_combinations, offset_max_corr_results):
        error = -max_corr
        # Offset is number of pixels to shift f2 to align with f1
        error_matrix[f1_idx, f2_idx] = error
        error_matrix[f2_idx, f1_idx] = error
        offset_matrix[f1_idx, f2_idx] = offset
        offset_matrix[f2_idx, f1_idx] = -offset  # reverse the offset for the second fragment
        
    print("error_matrix:")
    for row in error_matrix:
        for col_val in row:
            print(f"{col_val:.2f}", end=" ")
        print()
    # Mask lower triangular part of error_matrix
    # mask = np.tril(np.ones_like(error_matrix, dtype=bool), k=-1)
    # error_matrix[mask] = 1e6
    # Create the stitched image by following the minimum error path
    running_offset = np.zeros(2)
    composite = fragments[0]
    f1_idx = 0
    est_offsets = {0: running_offset.copy()}
    placed_fragments = {0}
    # Gather all absolute offsets for each fragment
    MAX_ITER = 4 * len(fragments)
    for _ in range(MAX_ITER):
        error_row = error_matrix[f1_idx].copy()
        # error_row[placed_fragments] = 1e6
        error_row[f1_idx] = 1e6
        
        min_indices = np.argsort(error_row)
        for f2_idx in min_indices:
            if f2_idx in placed_fragments:
                continue
            f2_idx = int(f2_idx)
            # placed_fragments.add(f1_idx)
            break
        error_val = error_row[f2_idx]
        if error_val >= 1e6:
            break
        print(f"{f1_idx=} {f2_idx=} {error_val=}")
        offset_f2_to_f1 = offset_matrix[f1_idx, f2_idx]
        
        running_offset += offset_f2_to_f1
        est_offsets[f2_idx] = running_offset.copy()
        print(f"{offset_f2_to_f1=} {running_offset=}")
        error_matrix[f1_idx, f2_idx] = 1e6
        # error_matrix[f2_idx, f1_idx] = 1e6
        f1_idx = f2_idx
        # placed_fragments.add(f1_idx)
    # assert placed_fragments[-1] == 0, f"{placed_fragments=}"
    # placed_fragments = placed_fragments[:-1]
        
    assert len(est_offsets) == len(fragments), f"{len(est_offsets)=} {len(fragments)=}"
    # assert len(placed_fragments) == len(fragments), f"{len(placed_fragments)=} {len(fragments)=}"
    # assert sorted(placed_fragments) == list(range(len(fragments))), f"{placed_fragments=}"
    
    # create a new composite image by combining all fragments
    min_offset = np.min(np.stack(list(est_offsets.values()), axis=0), axis=0)
    max_offset = np.max(np.stack(list(est_offsets.values()), axis=0), axis=0)
    max_image_size = np.max(np.stack([fragment.shape[:2] for fragment in fragments], axis=0), axis=0)
    num_channels = fragments[0].shape[-1]
    composite_shape = list(map(int, list(max_offset - min_offset + max_image_size))) + [num_channels]
    print(f"{composite_shape=}")
    composite = np.zeros(composite_shape)
    counts = np.zeros(composite_shape)
    for iter_num, (f1_idx, offset) in tqdm(enumerate(est_offsets.items()), total=len(est_offsets), desc="Stitching fragments"):
        image_fragment = fragments[f1_idx]
        offset = offset - min_offset
        start_r = int(offset[0])
        start_c = int(offset[1])
        end_r = int(start_r + image_fragment.shape[0])
        end_c = int(start_c + image_fragment.shape[1])
        if iter_num == 0:
            composite[start_r:end_r, start_c:end_c] += image_fragment
            counts[start_r:end_r, start_c:end_c] += 1
        else:
            # Fine-tune the offset
            additional_offset, _ = phase_correlation(composite[start_r:end_r, start_c:end_c], image_fragment, downsample_factor=1, use_skimage=use_skimage, full_convolution=False)
            offset += additional_offset
            start_r = int(offset[0])
            start_c = int(offset[1])
            end_r = int(start_r + image_fragment.shape[0])
            end_c = int(start_c + image_fragment.shape[1])
            if start_r < 0:
                image_fragment = image_fragment[-start_r:]
                start_r = 0
            if start_c < 0:
                image_fragment = image_fragment[:, -start_c:]
                start_c = 0
            if end_r > composite.shape[0]:
                image_fragment = image_fragment[:composite.shape[0] - end_r]
                end_r = composite.shape[0]
            if end_c > composite.shape[1]:
                image_fragment = image_fragment[:, :composite.shape[1] - end_c]
                end_c = composite.shape[1]
            composite[start_r:end_r, start_c:end_c] += image_fragment
            counts[start_r:end_r, start_c:end_c] += 1
    counts[counts == 0] = 1
    composite /= counts
        
    return composite, est_offsets


def new_stitch_images(fragments, downsample_factor=1, use_skimage=False):
    """Stitch a list of fragments together."""
    # Downsample the fragments
    downsampled_fragments = Parallel(n_jobs=-1)(delayed(skimage.transform.resize)(fragment, (fragment.shape[0] // downsample_factor, fragment.shape[1] // downsample_factor), order=1, anti_aliasing=True) \
        for fragment in tqdm(fragments, desc="Downsampling fragments"))
    assert len(fragments) == len(downsampled_fragments)
    



if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    image_paths = glob.glob("image_dir/*.png")
    image_paths = natsort.natsorted(image_paths)
    fragments = [np.array(Image.open(path)) / 255.0 for path in image_paths] #[:10]
    # random.shuffle(fragments)
    
    # Downsample the fragments
    downsample_factor = 4
    order = 1
    stitched, est_offsets = stitch_images(fragments, downsample_factor=downsample_factor, order=order, use_skimage=False)

    # print("Estimated Offsets:")
    # for (idx, off) in est_offsets.items():
    #     print(f"Fragment {idx}: {off}")

    # Show the stitched image
    plt.figure(figsize=(8, 8))
    plt.imshow(stitched)
    plt.title("Stitched Image")
    # plt.axis('off')
    plt.show()
    skimage.io.imsave("stitched.png", (stitched * 255).astype(np.uint8))
    
    
# if __name__ == '__main__':
#     # For demonstration purposes, we'll simulate a set of fragments from an image.
#     # In practice, you would load real images (for example, using imageio or OpenCV)
#     import matplotlib.pyplot as plt

#     # Create a synthetic test image
#     base_image = skimage.color.rgb2gray(skimage.data.astronaut())
#     # base_image = skimage.data.camera()
#     # rng = np.random.RandomState(42)
#     # base_image = rng.randint(0, 255, (200, 200)).astype(np.uint8)

#     # Generate fragments by cropping overlapping regions
#     fragments = []
#     # Offsets for the synthetic fragments (simulated translations)
#     image_size = min(base_image.shape)
#     skip_size = image_size // 8
#     y_offsets = range(0, image_size, skip_size)
#     x_offsets = range(0, image_size, skip_size)
#     true_offsets = [np.array([y_offset, x_offset]) for y_offset in y_offsets for x_offset in x_offsets]
#     # Shuffle the offsets
#     # rng = np.random.RandomState(42)
#     # rng.shuffle(true_offsets)
#     np.random.shuffle(true_offsets)
#     fragment_size = image_size // 3
#     assert skip_size < fragment_size
#     print(f"{skip_size=} {fragment_size=}")

#     for offset in true_offsets:
#         r_offset, c_offset = offset
#         # Define a bounding box for the fragment
#         start_r = max(0, 0 + r_offset)
#         start_c = max(0, 0 + c_offset)
#         end_r = min(image_size, fragment_size + r_offset)
#         end_c = min(image_size, fragment_size + c_offset)
#         fragment = base_image[start_r:end_r, start_c:end_c].copy()
#         fragments.append(fragment.astype(np.float32))
        
#     # # Visualize the fragments
#     # plt.figure(figsize=(10, 10))
#     # for i, fragment in enumerate(fragments):
#     #     plt.subplot(1, len(fragments), i+1)
#     #     plt.imshow(fragment, cmap='gray')
#     #     plt.title(f"Fragment {i}")
#     #     # plt.axis('off')
#     # plt.show()
#     # Stitch fragments together using our defined functions
#     stitched, est_offsets = stitch_images(fragments)

#     print("Estimated Offsets:")
#     for (idx, off), (true_idx, true_off) in zip(est_offsets.items(), enumerate(true_offsets)):
#         print(f"Fragment {idx}: {off}, True {true_idx}: {true_off}")

#     # Show the stitched image
#     plt.figure(figsize=(8, 8))
#     plt.subplot(1, 2, 1)
#     plt.imshow(base_image, cmap='gray')
#     plt.title("Base Image")
#     plt.subplot(1, 2, 2)
#     plt.imshow(stitched, cmap='gray')
#     plt.title("Stitched Image")
#     # plt.axis('off')
#     plt.show()