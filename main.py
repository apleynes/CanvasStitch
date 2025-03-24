#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
from skimage.registration import phase_cross_correlation

def pad_image(image, pad_shape):
    left_pad = (pad_shape - image.shape) // 2
    right_pad = pad_shape - image.shape - left_pad
    return np.pad(image, ((left_pad[0], right_pad[0]), (left_pad[1], right_pad[1])), mode='constant')

def phase_correlation(image_a, image_b):
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
    # Compute FFTs of both images.
    
    pad_shape = np.array([max(image_a.shape[0], image_b.shape[0]), max(image_a.shape[1], image_b.shape[1])])
    # Pad images to the same size, centered on the middle of the output shape
    image_a = pad_image(image_a, pad_shape)
    image_b = pad_image(image_b, pad_shape)
    
    fft_shape = 2 * pad_shape
    
    F_a = fft2(image_a, fft_shape)
    F_b = fft2(image_b, fft_shape)
    # Compute the cross-power spectrum
    cross_power = F_a * np.conj(F_b)
    # Normalize to get only phase information (add a small constant to prevent division by zero)
    cross_power /= (np.abs(cross_power) + 1e-8)
    # Inverse FFT to obtain correlation
    corr = ifft2(cross_power)
    # For ease of peak finding, shift the zero-frequency component to the center
    corr = fftshift(corr)

    # Find peak position in correlation
    max_pos = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    midpoints = np.array(corr.shape) // 2
    shift = np.array(max_pos) - midpoints
    return shift

def stitch_images(image_list):
    """
    Given a list of image fragments (assumed to be 2D numpy arrays),
    use phase correlation to estimate their relative positions and 
    output a stitched larger canvas.

    Assumptions:
      - All images are at the same scale and perspective.
      - Each image has sufficient overlap with at least one other image.

    Returns:
      canvas: the stitched image.
      offsets: a dictionary mapping image index to its top-left offset in the canvas.
    """
    # Initialize offsets with the first image placed at (0,0)
    offsets = {0: np.array([0, 0])}
    placed = {0}
    remaining = set(range(1, len(image_list)))

    # Greedy placement: For each unplaced image, try to find an already placed image 
    # with which it overlaps, then compute the relative translation.
    while remaining:
        progress = False
        to_be_removed = []
        for j in placed:
            for i in list(remaining):
                placed_index_errors = {}
                # For simplicity, assume the whole images overlap sufficiently.
                # In practice, one might need to select overlapping regions.
                try:
                    # Calculate the translation; note that phase_correlation returns the
                    # shift to align image_j to image_i. Therefore, we have:
                    #    image_i(x, y) ~ image_j(x + shift_y, y + shift_x)
                    image_a = image_list[j]
                    image_b = image_list[i]
                    pad_shape = np.array([max(image_a.shape[0], image_b.shape[0]), max(image_a.shape[1], image_b.shape[1])])
                    image_a = pad_image(image_a, pad_shape)
                    image_b = pad_image(image_b, pad_shape)
                    shift, error, diffphase = phase_cross_correlation(image_a, image_b)
                    # shift = phase_correlation(image_list[j], image_list[i])
                    # Compute absolute position of image i from image j
                    progress = True
                    placed_index_errors[i] = error
                    break  # Break out of inner loop once image i is placed
                except Exception as e:
                    # In cases where phase correlation fails (possibly due to no overlap),
                    # the algorithm will try another pair
                    continue
             
            # Get the minimum error index
            min_error = min(placed_index_errors.values())
            for key, value in placed_index_errors.items():
                if value == min_error:
                    min_error_index = key
                    break
            offsets[i] = offsets[min_error_index] - shift
            placed.add(i)
            to_be_removed.append(i)
                
        for i in to_be_removed:
            remaining.remove(i)
        if not progress:
            raise ValueError("Could not place all images; check overlaps between fragments.")

    # Estimate full canvas size from determined offsets and image dimensions.
    # Compute the min and max offsets.
    all_offsets = np.array(list(offsets.values()))
    min_offset = all_offsets.min(axis=0)
    max_offset = all_offsets.max(axis=0)
    # Use the size of fragments (assuming all images have same shape)
    frag_shape = np.array(image_list[0].shape)
    canvas_shape = (max_offset - min_offset) + frag_shape
    canvas_shape = canvas_shape.astype(int)

    # Create canvas and composite images.
    # For simplicity, we assume non-overlapping images. If overlaps occur,
    # one could average the pixel values.
    canvas = np.zeros(canvas_shape, dtype=image_list[0].dtype)
    for idx, offset in offsets.items():
        top_left = (offset - min_offset).astype(int)
        r0, c0 = top_left
        r1, c1 = r0 + image_list[idx].shape[0], c0 + image_list[idx].shape[1]
        # Paste image fragment into the canvas.
        canvas[r0:r1, c0:c1] = image_list[idx]

    return canvas, offsets

if __name__ == '__main__':
    # For demonstration purposes, we'll simulate a set of fragments from an image.
    # In practice, you would load real images (for example, using imageio or OpenCV)
    import matplotlib.pyplot as plt

    # Create a synthetic test image
    base_image = skimage.color.rgb2gray(skimage.data.astronaut())
    # rng = np.random.RandomState(42)
    # base_image = rng.randint(0, 255, (200, 200)).astype(np.uint8)

    # Generate fragments by cropping overlapping regions
    fragments = []
    # Offsets for the synthetic fragments (simulated translations)
    image_size = min(base_image.shape)
    skip_size = image_size // 8
    y_offsets = range(0, image_size, skip_size)
    x_offsets = range(0, image_size, skip_size)
    true_offsets = [np.array([y_offset, x_offset]) for y_offset in y_offsets for x_offset in x_offsets]
    fragment_size = image_size // 2
    assert skip_size < fragment_size

    for offset in true_offsets:
        r_offset, c_offset = offset
        # Define a bounding box for the fragment
        start_r = max(0, 0 + r_offset)
        start_c = max(0, 0 + c_offset)
        end_r = min(image_size, fragment_size + r_offset)
        end_c = min(image_size, fragment_size + c_offset)
        fragment = base_image[start_r:end_r, start_c:end_c].copy()
        fragments.append(fragment.astype(np.float32))
        
    # # Visualize the fragments
    # plt.figure(figsize=(10, 10))
    # for i, fragment in enumerate(fragments):
    #     plt.subplot(1, len(fragments), i+1)
    #     plt.imshow(fragment, cmap='gray')
    #     plt.title(f"Fragment {i}")
    #     # plt.axis('off')
    # plt.show()
    # Stitch fragments together using our defined functions
    stitched, est_offsets = stitch_images(fragments)

    print("Estimated Offsets:")
    for (idx, off), (true_idx, true_off) in zip(est_offsets.items(), enumerate(true_offsets)):
        print(f"Fragment {idx}: {off}, True {true_idx}: {true_off}")

    # Show the stitched image
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(base_image, cmap='gray')
    plt.title("Base Image")
    plt.subplot(1, 2, 2)
    plt.imshow(stitched, cmap='gray')
    plt.title("Stitched Image")
    # plt.axis('off')
    plt.show()
