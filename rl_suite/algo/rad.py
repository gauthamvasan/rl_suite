import torch

def random_augment(images, rad_height, rad_width):
    n, c, h, w = images.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_images = torch.empty((n, c, _h, _w), device=images.device).float()
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped_images[i][:] = image[:, h11:h11 + _h, w11:w11 + _w]
    return