# Canvas Stitch

A tool for stitching together a collection of images into a single image.

Intended to be used for stitching together a collection of zoomed-in canvas fragments (e.g., screenshots of a canvas in Miro, Google Draw, etc.) into a single image.

Can only handle image fragments with the same zoom level.

For example, you need to take screenshots of a canvas at the same zoom level to be able to stitch them together.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python canvas_stitch.py [-h] [--output_path OUTPUT_PATH] [--downsample_factor DOWNSAMPLE_FACTOR] [--order ORDER] [--weight_edges] [--max_recursion_count MAX_RECURSION_COUNT] [--debug]
                        image_dir

positional arguments:
  image_dir             Path to the directory containing the images to stitch.

options:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Path to save the stitched image. Defaults to 'stitched.png'.
  --downsample_factor DOWNSAMPLE_FACTOR
                        Downsample factor for the fragments. Defaults to 4.
  --order ORDER         Interpolation order for the fragments. Defaults to 0.
  --weight_edges        Whether to weight the edges of the fragments. Defaults to False.
  --max_recursion_count MAX_RECURSION_COUNT
                        Maximum recursion count. Defaults to the number of images if unspecified.
  --debug               Whether to print debug information. Defaults to False.
```

Only `image_dir` is required. All other arguments are optional.

Tested with Python 3.11.

## License

MIT License

## Disclaimer

This is a simple tool that I wrote for my own use. I'm sharing it here in case it's useful for others.

Use the code at your own risk. No guarantees are made about the quality or correctness of the code. No support is provided.





