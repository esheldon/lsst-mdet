"""
small shared utilities
"""


def get_stamp(image, x, y, stamp_size):
    """
    Extract a postage stamp from the input image.   Returned stamps are square.
    If the stamp hits an edge, IndexError is raised.

    Parameters
    ----------
    image: array
        Array from which to extract a stamp
    x: float or int
        x position in the array
    y: float or int
        y position in the array
    stamp_size: int
        The extracted stamp will have size [stamp_size, stamp_size]

    Returns
    -------
    stamp, xstart, ystart

    stamp: array
        Extracted stamp
    xstart: int
        The start x position
    ystart: int
        The start y position
    """

    assert stamp_size % 2 != 0, f'stamp size should be odd, got {stamp_size}'

    imny, imnx = image.shape
    xstart, xend = _get_bound(x, stamp_size, imsize=imnx)
    ystart, yend = _get_bound(y, stamp_size, imsize=imny)

    stamp = image[ystart:yend, xstart:xend]
    if stamp.shape[0] != stamp_size or stamp.shape[1] != stamp_size:
        raise IndexError(
            f'expected shape [{stamp_size}, {stamp_size}] '
            f'but got {stamp.shape}'
        )
    return stamp, xstart, ystart


def _get_bound(x, stamp_size, imsize):
    """
    Get the bound in one dimension for a requested stamp

    Parameters
    ----------
    x: float or int
        x position in the array
    image: array
        Array from which to extract a stamp
    y: float or int
        y position in the array
    stamp_size: int
        The extracted stamp will have size [stamp_size, stamp_size]

    """
    rx = round(x)
    xstart = rx - (stamp_size - 1) // 2
    xend = rx + (stamp_size - 1) // 2 + 1

    if xstart < 0:
        raise IndexError('out of bounds')
    if xend > imsize + 1:
        raise IndexError('out of bounds')

    return xstart, xend
