import unittest
import skimage.io
import numpy as np
from PIL import Image
from canvas_stitch import get_image_files_in_dir, stitch_images


class TestCanvasStitch(unittest.TestCase):
    def setUp(self):
        self.image_dir = "test_images"
        self.expected_image = Image.open("test_output.png")
        self.expected_image = np.array(self.expected_image) / 255.0

    def test_stitch_images(self):
        image_paths = get_image_files_in_dir(self.image_dir)
        fragments = [np.array(Image.open(path)) / 255.0 for path in image_paths]
        stitched = stitch_images(fragments, downsample_factor=4, order=0, weight_edges=False, max_recursion_count=len(fragments))
        self.assertEqual(stitched.shape, self.expected_image.shape)
        self.assertTrue(np.allclose(stitched, self.expected_image))

if __name__ == "__main__":
    unittest.main()
