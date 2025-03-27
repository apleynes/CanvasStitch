import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.fft import fft2, ifft2, fftshift, rfft2, irfft2, ifftshift
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
        padded_image = np.pad(
            image,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded_image = np.pad(
            image,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            mode="constant",
            constant_values=0,
        )
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

    new_composite = np.pad(
        composite,
        ((top_pad, bottom_pad), (left_pad, right_pad)),
        mode="constant",
        constant_values=0,
    )
    counts = np.zeros(new_composite.shape)
    counts[
        top_pad : top_pad + composite.shape[0], left_pad : left_pad + composite.shape[1]
    ] = 1
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



def weight_edge_image_ft(image_ft, power=1, rfft=False):
    shape = image_ft.shape
    if not rfft:
        grid = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing='ij')
        grid = np.stack(grid, axis=-1)
        dists = np.linalg.norm(grid, axis=-1)
        dists = ifftshift(dists)
    else:
        # TODO: Fix this. Only the last axis is halved in size by default behavior of rfftn.
        coords = []
        for i in range(len(shape)):
            if i == len(shape) - 1:
                coords.append(np.linspace(0, 1, shape[i]))
                continue
            c = np.linspace(-1, 1, shape[i])
            coords.append(np.roll(c, shift=-np.where(np.isclose(c.flatten(), 0))[0]))
        grid = np.meshgrid(*coords, indexing='ij')
        grid = np.stack(grid, axis=-1)
        dists = np.linalg.norm(grid, axis=-1)
    weights = dists / np.max(dists)
    return (weights ** power) * image_ft
    

phase_correlation_cache = {}
def phase_correlation(
    image_a,
    image_b,
    correlation_threshold=0.0,
    distance_threshold=1.0,
    downsample_factor=1,
    use_skimage=False,
    full_convolution=True,
    weight_edges=True,
    use_rfft=True,
):
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
            
    dtype_max = 1.0
    if image_a.dtype == np.uint8:
        dtype_max = 255.0
    elif image_a.dtype == np.float32:
        dtype_max = 1.0
    elif image_a.dtype == np.float64:
        dtype_max = 1.0
    elif image_a.dtype == np.int16:
        dtype_max = 32767.0
    elif image_a.dtype == np.int32:
        dtype_max = 2147483647.0
    elif image_a.dtype == np.int64:
        dtype_max = 9223372036854775807.0
    elif image_a.dtype == np.uint16:
        dtype_max = 65535.0
    elif image_a.dtype == np.uint32:
        dtype_max = 4294967295.0
    elif image_a.dtype == np.uint64:
        dtype_max = 18446744073709551615.0

    # Compute FFTs of both images.
    pad_shape = np.array(
        [
            max(image_a.shape[0], image_b.shape[0]),
            max(image_a.shape[1], image_b.shape[1]),
        ]
    )

    # Convert to grayscale if needed
    if image_a.ndim > 2:    
        if image_a.shape[2] == 4:
            image_a = rgba2gray(image_a)
        elif image_a.shape[2] == 3:
            image_a = skimage.color.rgb2gray(image_a)
        else:
            raise ValueError(f"Unexpected number of channels: {image_a.shape[2]}")
    if image_b.ndim > 2:
        if image_b.shape[2] == 4:
            image_b = rgba2gray(image_b)
        elif image_b.shape[2] == 3:
            image_b = skimage.color.rgb2gray(image_b)
        else:
            raise ValueError(f"Unexpected number of channels: {image_b.shape[2]}")
      

    if use_skimage:
        # Pad images to the same size
        image_a, _ = pad_image(image_a, pad_shape)
        image_b, _ = pad_image(image_b, pad_shape)
        shift, error, phasediff = phase_cross_correlation(
            image_a,
            image_b,
            disambiguate=full_convolution,
            overlap_ratio=0.1,
            normalization=None,
        )
        shift = shift * downsample_factor
        return shift, -error - phasediff

    fft_shape = 2 * pad_shape + 1 if full_convolution else pad_shape
    if use_rfft:
        # fft_shape = pad_shape + 1 if full_convolution else (pad_shape + 1) // 2
        F_a = rfft2(image_a, fft_shape, norm='backward')
        F_b = rfft2(image_b, fft_shape, norm='backward')
    else:
        F_a = fft2(image_a, fft_shape, norm='backward')
        F_b = fft2(image_b, fft_shape, norm='backward')
    
    if weight_edges:
        F_a = weight_edge_image_ft(F_a, rfft=use_rfft)
        F_b = weight_edge_image_ft(F_b, rfft=use_rfft)

    # Compute the cross-power spectrum
    cross_power = F_a * np.conj(F_b)
    # Normalize to get only phase information (add a small constant to prevent division by zero)
    cross_power /= np.abs(cross_power) + 1e-8
    # Inverse FFT to obtain correlation
    if use_rfft:
        corr = irfft2(cross_power, fft_shape, norm='backward')
    else:
        corr = ifft2(cross_power, fft_shape, norm='backward')
    # For ease of peak finding, shift the zero-frequency component to the center
    corr = fftshift(corr)

    if distance_threshold > 0 and distance_threshold < 1:
        # Mask based on L-infinity distance
        fraction_size = list(map(int, np.array(corr.shape) * distance_threshold))
        midpoints = np.array(corr.shape) // 2
        corr = corr[
            (midpoints[0] - fraction_size[0] // 2) : (
                midpoints[0] + fraction_size[0] // 2
            ),
            (midpoints[1] - fraction_size[1] // 2) : (
                midpoints[1] + fraction_size[1] // 2
            ),
        ]
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
    """
    Convert an RGBA image to grayscale.

    Parameters
    ----------
    image : array_like
        Input image of shape (H, W, 4).

    Returns
    -------
    image : array_like
        Output image of shape (H, W).
    """
    im = skimage.color.rgba2rgb(image)
    return skimage.color.rgb2gray(im)


def stitch_images(
    fragments, downsample_factor=1, order=1, use_skimage=False, debug=False
):
    """Stitch a list of fragments together."""
    # Crop the fragments to be divisible by downsample_factor
    fragments = [
        fragment[
            : (fragment.shape[0] // downsample_factor) * downsample_factor,
            : (fragment.shape[1] // downsample_factor) * downsample_factor,
        ]
        for fragment in fragments
    ]

    # Downsample the fragments
    downsampled_fragments = Parallel(n_jobs=-1)(
        delayed(skimage.transform.resize)(
            fragment,
            (
                fragment.shape[0] // downsample_factor,
                fragment.shape[1] // downsample_factor,
            ),
            order=order,
            anti_aliasing=True if order > 0 else False,
        )
        for fragment in tqdm(fragments, desc="Downsampling fragments")
    )
    assert len(fragments) == len(downsampled_fragments)
    # Iterate through the fragments and compute the offset for each fragment against each other fragment
    error_matrix = 1e6 * np.ones((len(fragments), len(fragments)))
    offset_matrix = np.zeros((len(fragments), len(fragments), 2))
    # Do in parallel with joblib
    f1_f2_idx_combinations = list(itertools.combinations(range(len(fragments)), 2))
    offset_max_corr_results = Parallel(n_jobs=-1)(
        delayed(phase_correlation)(
            downsampled_fragments[f1_idx],
            downsampled_fragments[f2_idx],
            downsample_factor=downsample_factor,
            use_skimage=use_skimage,
            full_convolution=True,
        )
        for f1_idx, f2_idx in tqdm(f1_f2_idx_combinations)
    )
    # Unpack the results
    for (f1_idx, f2_idx), (offset, max_corr) in zip(
        f1_f2_idx_combinations, offset_max_corr_results
    ):
        error = -max_corr
        # Offset is number of pixels to shift f2 to align with f1
        error_matrix[f1_idx, f2_idx] = error
        error_matrix[f2_idx, f1_idx] = error
        offset_matrix[f1_idx, f2_idx] = offset
        offset_matrix[
            f2_idx, f1_idx
        ] = -offset  # reverse the offset for the second fragment

    if debug:
        print("error_matrix:")
        for row in error_matrix:
            for col_val in row:
                print(f"{col_val:.2f}", end=" ")
            print()
    # Create the stitched image by following the minimum error path
    running_offset = np.zeros(2)
    composite = fragments[0]
    f1_idx = 0
    est_offsets = {0: running_offset.copy()}
    placed_fragments = {0}
    traced_paths = set()
    # Gather all absolute offsets for each fragment
    MAX_ITER = 4 * len(fragments)
    for _ in range(MAX_ITER):
        error_row = error_matrix[f1_idx].copy()
        # error_row[placed_fragments] = 1e6
        error_row[f1_idx] = 1e6

        min_indices = np.argsort(error_row)
        for f2_idx in min_indices:  # TODO: Need to figure this out. Something funny going on since this works but not when I simplify it.
            if f2_idx in placed_fragments:
                continue
            f2_idx = int(f2_idx)
            if (f1_idx, f2_idx) in traced_paths:
                continue
            # placed_fragments.add(f1_idx)
            traced_paths.add((f1_idx, f2_idx))
            break
        error_val = error_row[f2_idx]
        if error_val >= 1e6:
            break
        if debug:
            print(f"{f1_idx=} {f2_idx=} {error_val=}")
        offset_f2_to_f1 = offset_matrix[f1_idx, f2_idx]

        running_offset += offset_f2_to_f1
        est_offsets[f2_idx] = running_offset.copy()
        if debug:
            print(f"{offset_f2_to_f1=} {running_offset=}")
        error_matrix[f1_idx, f2_idx] = 1e6  # block forward path
        error_matrix[f2_idx, f1_idx] = 1e6  # block backward path
        f1_idx = f2_idx

    assert len(est_offsets) == len(fragments), f"{len(est_offsets)=} {len(fragments)=}"

    # create a new composite image by combining all fragments
    min_offset = np.min(np.stack(list(est_offsets.values()), axis=0), axis=0)
    max_offset = np.max(np.stack(list(est_offsets.values()), axis=0), axis=0)
    max_image_size = np.max(
        np.stack([fragment.shape[:2] for fragment in fragments], axis=0), axis=0
    )
    num_channels = fragments[0].shape[-1]
    composite_shape = list(map(int, list(max_offset - min_offset + max_image_size))) + [
        num_channels
    ]
    print(f"{composite_shape=}")
    composite = np.zeros(composite_shape)
    counts = np.zeros(composite_shape)
    for iter_num, (f1_idx, offset) in tqdm(
        enumerate(est_offsets.items()),
        total=len(est_offsets),
        desc="Stitching fragments",
    ):
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
            additional_offset, _ = phase_correlation(
                composite[start_r:end_r, start_c:end_c],
                image_fragment,
                downsample_factor=1,
                use_skimage=use_skimage,
                full_convolution=False,
            )
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
                image_fragment = image_fragment[: composite.shape[0] - end_r]
                end_r = composite.shape[0]
            if end_c > composite.shape[1]:
                image_fragment = image_fragment[:, : composite.shape[1] - end_c]
                end_c = composite.shape[1]
            composite[start_r:end_r, start_c:end_c] += image_fragment
            counts[start_r:end_r, start_c:end_c] += 1
    counts[counts == 0] = 1
    composite /= counts

    return composite, est_offsets


def stitch_two_images(image_a, image_b, offset, fine_tune_offset=True):
    """Combine two fragments with an offset."""
    # Offset is number of pixels to shift f2 to align with f1
    # Offset is the absolute offset of f2 relative to composite
    # Calculate the expected combined image size using the offset
    
    # Convert to floats to avoid overflow
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)
    
    image_b_shape = image_b.shape
    
    # Fine-tune the offset using the overlapping region
    if fine_tune_offset:
        # Find the overlapping region
        a_start_r = max(0, int(offset[0]))
        a_start_c = max(0, int(offset[1]))
        a_end_r = min(image_a.shape[0], int(offset[0] + image_b_shape[0]))
        a_end_c = min(image_a.shape[1], int(offset[1] + image_b_shape[1]))

        b_start_r = max(0, int(-offset[0]))
        b_start_c = max(0, int(-offset[1]))
        b_end_r = min(image_b_shape[0], int(image_a.shape[0] - offset[0]))
        b_end_c = min(image_b_shape[1], int(image_a.shape[1] - offset[1]))
        
        if (a_end_r - a_start_r) > 0 and (a_end_c - a_start_c) > 0 and (b_end_r - b_start_r) > 0 and (b_end_c - b_start_c) > 0:
            additional_offset, _ = phase_correlation(
                image_a[a_start_r:a_end_r, a_start_c:a_end_c],
                image_b[b_start_r:b_end_r, b_start_c:b_end_c],
                downsample_factor=1,
                use_skimage=False,
                full_convolution=False,
                weight_edges=False,
            )
            offset += additional_offset

    start_y = offset[0]
    start_x = offset[1]
    end_y = start_y + image_b_shape[0]
    end_x = start_x + image_b_shape[1]
    top_pad = int(-min(0, start_y))
    left_pad = int(-min(0, start_x))
    bottom_pad = int(max(0, end_y - image_a.shape[0]))
    right_pad = int(max(0, end_x - image_a.shape[1]))

    if image_a.ndim == 2:
        pad_values = ((top_pad, bottom_pad), (left_pad, right_pad))
    else:
        pad_values = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    new_composite = np.pad(
        image_a,
        pad_values,
        mode="constant",
        constant_values=0,
    )
    counts = np.zeros(new_composite.shape, dtype=int)
    # counts[
    #     top_pad : top_pad + image_a.shape[0], left_pad : left_pad + image_a.shape[1], :
    # ] = 1
    counts[new_composite > 0] = 1  # To also include pixels already in the composite
    # Place new fragment in the new composite
    start_r = int(top_pad + start_y)
    start_c = int(left_pad + start_x)
    end_r = int(start_r + image_b_shape[0])
    end_c = int(start_c + image_b_shape[1])
    new_composite[start_r:end_r, start_c:end_c, :] += image_b
    counts[start_r:end_r, start_c:end_c, :] += (image_b > 0).astype(int)
    counts[counts == 0] = 1
    new_composite = new_composite / counts
    return new_composite


MAX_RECURSION_COUNT = 10
def new_stitch_images(
    fragments, downsample_factor=1, order=1, use_skimage=False, debug=False, weight_edges=False, recursion_count=0
):
    """Stitch a list of fragments together."""
    print(f"{recursion_count=}")
    if recursion_count > MAX_RECURSION_COUNT:
        return fragments[0]
    if len(fragments) == 1:
        return fragments[0]
    if downsample_factor > 1:
        # Crop the fragments to be divisible by downsample_factor
        fragments = [
            fragment[
                : (fragment.shape[0] // downsample_factor) * downsample_factor,
                : (fragment.shape[1] // downsample_factor) * downsample_factor,
            ]
            for fragment in fragments
        ]

        # Downsample the fragments
        downsampled_fragments = Parallel(n_jobs=-1)(
            delayed(skimage.transform.resize)(
                fragment,
                (
                    fragment.shape[0] // downsample_factor,
                    fragment.shape[1] // downsample_factor,
                ),
                order=order,
                anti_aliasing=True if order > 0 else False,
            )
            for fragment in tqdm(fragments, desc="Downsampling fragments")
        )
        assert len(fragments) == len(downsampled_fragments)
    else:
        downsampled_fragments = fragments
    # Iterate through the fragments and compute the offset for each fragment against each other fragment
    cross_correlation_matrix = np.zeros((len(fragments), len(fragments)))
    offset_matrix = np.zeros((len(fragments), len(fragments), 2))
    # Do in parallel with joblib
    f1_f2_idx_combinations = list(itertools.combinations(range(len(fragments)), 2))
    offset_max_corr_results = Parallel(n_jobs=-1)(
        delayed(phase_correlation)(
            downsampled_fragments[f1_idx],
            downsampled_fragments[f2_idx],
            downsample_factor=downsample_factor,
            use_skimage=use_skimage,
            full_convolution=True,
            weight_edges=weight_edges,
        )
        for f1_idx, f2_idx in tqdm(f1_f2_idx_combinations, desc="Computing cross correlation")
    )
    # Unpack the results
    for (f1_idx, f2_idx), (offset, max_corr) in zip(
        f1_f2_idx_combinations, offset_max_corr_results
    ):
        cross_correlation_matrix[f1_idx, f2_idx] = max_corr
        cross_correlation_matrix[f2_idx, f1_idx] = max_corr
        offset_matrix[f1_idx, f2_idx] = offset
        offset_matrix[
            f2_idx, f1_idx
        ] = -offset  # reverse the offset for the second fragment
        
    new_fragments = []
    placed_fragments = set()
    for f1_idx in tqdm(range(len(fragments)), desc="Stitching fragments"):
        if f1_idx in placed_fragments:
            continue
        f2_idx = np.argmax(cross_correlation_matrix[f1_idx])
        if f2_idx in placed_fragments:  # Completely skip if the fragment it would match with has already been taken, revisit it on later recursion
            new_fragments.append(fragments[f1_idx])
            continue
        offset = offset_matrix[f1_idx, f2_idx]
        # # Fine-tune the offset
        # if downsample_factor > 1:
        #     additional_offset, _ = phase_correlation(
        #         fragments[f1_idx],
        #         fragments[f2_idx],
        #         downsample_factor=1,
        #         use_skimage=use_skimage,
        #         full_convolution=False,
        #         weight_edges=weight_edges,
        #     )
        #     offset += additional_offset
        # print(f"{offset=}")
        composite = stitch_two_images(fragments[f1_idx], fragments[f2_idx], offset)
        new_fragments.append(composite)
        placed_fragments.add(f1_idx)
        placed_fragments.add(f2_idx)


    print(f"{len(fragments)=}")
    print(f"{len(new_fragments)=}")
    # Recursively stitch the new fragments together
    return new_stitch_images(new_fragments, downsample_factor=downsample_factor, order=order, use_skimage=use_skimage, debug=debug, recursion_count=recursion_count + 1)


# ! Currently only works if the fragments sequentially along a smooth path since the stitching is done by following the minimum error path.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    image_paths = glob.glob("test_images/*.png")
    image_paths = natsort.natsorted(image_paths)
    fragments = [np.array(Image.open(path)) / 255.0 for path in image_paths]#[:10]
    random.shuffle(fragments)
    
    # Convert fragments to RGB if RGBA
    fragments = [fragment[:, :, :3] if fragment.shape[2] == 4 else fragment for fragment in fragments]
    
    # # Preview fragments
    # plt.figure(figsize=(8, 8))
    # for i, fragment in enumerate(fragments):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(fragment)
    #     plt.title(f"Fragment {i}")
    #     plt.axis('off')
    # plt.show()

    # Downsample the fragments
    downsample_factor = 8
    order = 1
    stitched = new_stitch_images(
        fragments, downsample_factor=downsample_factor, order=order, use_skimage=False, weight_edges=True
    )

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
