
import os
from joblib import Parallel, delayed
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, rfft2, irfft2, ifftshift
import skimage
import skimage.io
import itertools
from tqdm import tqdm
import natsort
import argparse
from PIL import Image



def weight_edge_image_ft(image_ft: np.ndarray, power: int = 1, rfft: bool = False) -> np.ndarray:
    """Apply edge weighting in frequency domain.
    
    Creates distance-based weighting mask to reduce boundary artifacts.
    
    Args:
        image_ft (np.ndarray): Complex FFT of image
        power (int): Exponent for distance weighting
        rfft (bool): Whether using real-valued FFT
        
    Returns:
        np.ndarray: Weighted frequency spectrum
    """
    shape = image_ft.shape
    if not rfft:
        grid = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
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
        grid = np.meshgrid(*coords, indexing="ij")
        grid = np.stack(grid, axis=-1)
        dists = np.linalg.norm(grid, axis=-1)
    weights = dists / np.max(dists)
    return (weights**power) * image_ft


def phase_correlation(
    image_a,
    image_b,
    correlation_threshold=0.0,
    distance_threshold=1.0,
    downsample_factor=1,
    full_convolution=True,
    weight_edges=True,
    use_rfft=True,
):
    """Compute relative translation between images using phase correlation.
    
    Args:
        image_a (np.ndarray): Reference image (grayscale or RGB)
        image_b (np.ndarray): Moving image to align with reference
        correlation_threshold (float): Minimum correlation value to consider
        distance_threshold (float): Limit search space to this fraction of image
        downsample_factor (int): Downsampling factor for alignment
        use_skimage (bool): Use scikit-image's implementation
        full_convolution (bool): Use full convolution rather than FFT padding
        weight_edges (bool): Apply frequency domain edge weighting
        use_rfft (bool): Use real-valued FFT for efficiency

    Returns:
        tuple: (shift vector, max correlation value)
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

    fft_shape = 2 * pad_shape + 1 if full_convolution else pad_shape
    if use_rfft:
        # fft_shape = pad_shape + 1 if full_convolution else (pad_shape + 1) // 2
        F_a = rfft2(image_a, fft_shape, norm="backward")
        F_b = rfft2(image_b, fft_shape, norm="backward")
    else:
        F_a = fft2(image_a, fft_shape, norm="backward")
        F_b = fft2(image_b, fft_shape, norm="backward")

    if weight_edges:
        F_a = weight_edge_image_ft(F_a, rfft=use_rfft)
        F_b = weight_edge_image_ft(F_b, rfft=use_rfft)

    # Compute the cross-power spectrum
    cross_power = F_a * np.conj(F_b)
    # Normalize to get only phase information (add a small constant to prevent division by zero)
    cross_power /= np.abs(cross_power) + 1e-8
    # Inverse FFT to obtain correlation
    if use_rfft:
        corr = irfft2(cross_power, fft_shape, norm="backward")
    else:
        corr = ifft2(cross_power, fft_shape, norm="backward")
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


def rgba2gray(image: np.ndarray) -> np.ndarray:
    """Convert RGBA image to grayscale with alpha handling.
    
    Args:
        image (np.ndarray): Input RGBA image (H, W, 4)
        
    Returns:
        np.ndarray: Grayscale image (H, W)
        
    Raises:
        ValueError: If input is not 4-channel
    """
    if image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got {image.shape} shape")
    return skimage.color.rgb2gray(skimage.color.rgba2rgb(image))


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

        if (
            (a_end_r - a_start_r) > 0
            and (a_end_c - a_start_c) > 0
            and (b_end_r - b_start_r) > 0
            and (b_end_c - b_start_c) > 0
        ):
            additional_offset, _ = phase_correlation(
                image_a[a_start_r:a_end_r, a_start_c:a_end_c],
                image_b[b_start_r:b_end_r, b_start_c:b_end_c],
                downsample_factor=1,
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
def stitch_images(
    fragments,
    downsample_factor=1,
    order=1,
    debug=False,
    weight_edges=False,
    recursion_count=0,
    max_recursion_count=MAX_RECURSION_COUNT,
):
    """Stitch a list of fragments together using recursive alignment.
    
    Args:
        fragments (list): List of image arrays to stitch together
        downsample_factor (int): Factor to downsample images for faster alignment
        order (int): Interpolation order for resizing
        debug (bool): Enable debug outputs
        weight_edges (bool): Apply edge weighting in frequency domain
        recursion_count (int): Current recursion depth
        max_recursion_count (int): Maximum recursion depth
    Returns:
        np.ndarray: Stitched composite image
        
    Raises:
        ValueError: If fragments list is empty
        RuntimeError: If maximum recursion depth is exceeded
    """
    if len(fragments) < 1:
        raise ValueError("Empty fragment list provided")
        
    if len(fragments) == 1:
        return fragments[0]
        
    if recursion_count > max_recursion_count:
        raise RuntimeError(
            f"Max recursion depth {max_recursion_count} reached. "
            "Check fragment alignment."
        )
        
    if debug:
        print(f"{recursion_count=}")
    
    if downsample_factor > 1:
        # Crop the fragments to be divisible by downsample_factor
        downsampled_fragments = [
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
            for fragment in tqdm(downsampled_fragments, desc="Downsampling fragments")
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
            full_convolution=True,
            weight_edges=weight_edges,
        )
        for f1_idx, f2_idx in tqdm(
            f1_f2_idx_combinations, desc="Computing cross correlation"
        )
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
        if (
            f2_idx in placed_fragments
        ):  # Completely skip if the fragment it would match with has already been taken, revisit it on later recursion
            new_fragments.append(fragments[f1_idx])
            continue
        offset = offset_matrix[f1_idx, f2_idx]
        composite = stitch_two_images(fragments[f1_idx], fragments[f2_idx], offset)
        new_fragments.append(composite)
        placed_fragments.add(f1_idx)
        placed_fragments.add(f2_idx)

    if debug:
        print(f"{len(fragments)=}")
        print(f"{len(new_fragments)=}")
    # Recursively stitch the new fragments together
    return stitch_images(
        new_fragments,
        downsample_factor=downsample_factor,
        order=order,
        debug=debug,
        recursion_count=recursion_count + 1,
        max_recursion_count=max_recursion_count,
    )



def get_image_files_in_dir(image_dir: str) -> list[str]:
    """Get all image files in a directory."""
    # Get all files in the directory
    files = os.listdir(image_dir)
    # Filter out non-image files
    image_paths = [os.path.join(image_dir, file) for file in files if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    image_paths = natsort.natsorted(image_paths)
    return image_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="Path to the directory containing the images to stitch.")
    parser.add_argument("--output_path", type=str, required=False, default="stitched.png", help="Path to save the stitched image. Defaults to 'stitched.png'.")
    parser.add_argument("--downsample_factor", type=int, required=False, default=4, help="Downsample factor for the fragments. Defaults to 4.")
    parser.add_argument("--order", type=int, required=False, default=0, help="Interpolation order for the fragments. Defaults to 0.")
    parser.add_argument("--weight_edges", action="store_true", required=False, default=False, help="Whether to weight the edges of the fragments. Defaults to False.")
    parser.add_argument("--max_recursion_count", type=int, required=False, default=None, help="Maximum recursion count. Defaults to the number of images if unspecified.")
    parser.add_argument("--debug", action="store_true", required=False, default=False, help="Whether to print debug information. Defaults to False.")
    args = parser.parse_args()
    
    image_paths = get_image_files_in_dir(args.image_dir)
    fragments = [np.array(Image.open(path)) / 255.0 for path in image_paths]
    max_recursion_count = len(fragments) if args.max_recursion_count is None else args.max_recursion_count

    # Downsample the fragments
    downsample_factor = args.downsample_factor
    order = args.order
    weight_edges = args.weight_edges
    debug = args.debug
    stitched = stitch_images(
        fragments,
        downsample_factor=downsample_factor,
        order=order,
        weight_edges=weight_edges,
        max_recursion_count=max_recursion_count,
        debug=debug,
    )

    # Save the stitched image
    Image.fromarray((stitched * 255).astype(np.uint8)).save(args.output_path)
