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
python canvas_stitch.py <image_dir> --output_path <output_path> --downsample_factor <downsample_factor> --order <order> --weight_edges --max_recursion_count <max_recursion_count> --debug
```

Only `image_dir` is required. All other arguments are optional.

Tested with Python 3.11.

## License

MIT License

## Disclaimer

This is a simple tool that I wrote for my own use. I'm sharing it here in case it's useful for others.

Use the code at your own risk. No guarantees are made about the quality or correctness of the code. No support is provided.





