from PIL import Image

def imresize(img, scale_factor, antialias=True):
    """
    Resizes the PIL image `img` by `scale_factor` using LANCZOS if antialias is True.
    """
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
    return img.resize(new_size, Image.LANCZOS if antialias else Image.NEAREST)
