import numpy as np
import nrrd
from tqdm import trange, tqdm
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import json
import matplotlib.pyplot as plt



def get_atlas(path):
    d, h = nrrd.read(path)
    return d


def contourify(stack):
    """
    Get the  contours of the regions in the atlas
    Very slow. Should be done once and the result saved

    Parameters
    ----------
    stack: numpy ndarray
        Atlas annotations

    Returns
    -------
    contoured: numpy ndarray
        Contoured atlas
    """

    contoured = np.zeros(stack.shape, dtype=stack.dtype)
    for ix in trange(stack.shape[2]):
        c_slice = stack[:, :, ix]
        structures_ix = np.unique(c_slice.reshape(-1))
        for c_struct in tqdm(structures_ix):
            if c_struct == 0:
                continue
            contours = find_contours(c_slice, c_struct - .5)
            for cnt in contours:
                rr, cc = polygon_perimeter(*cnt.T, c_slice.shape)
                contoured[rr, cc, ix] = c_struct
    return contoured


def read_ontology(path):
    with open(path, 'r') as f:
        an = json.load(f)
    return an['msg']


def id_colors(onto, colors={}):
    """
    From an ontology return the colors corresponding to a structure

    Parameters
    ----------
    onto: list
        Ontology as returned from ::py:func: `read_ontology`
    colors: dict
        Used because function is recursive. Should be left at default when called from outside

    Returns
    -------
    colors: dict
    """
    for s in onto:
        colors[s['id']] = s['color_hex_triplet']
        id_colors(s['children'])
    return colors


def color_atlas(atlas, colors):
    ids = np.unique(atlas)
    c_atlas = np.zeros(atlas.shape + (3,), dtype=np.uint8)
    for ix in tqdm(ids):
        if ix == 0:
            continue
        try:
            color = int(colors[ix], 16)
            r = color >> 16
            g = (color - (r << 16)) >> 8
            b = color - (r << 16) - (g << 8)
        except KeyError:
            r, g, b = 255, 255, 255
            print(ix)
        gi = atlas == ix
        c_atlas[gi, 0] = r
        c_atlas[gi, 1] = g
        c_atlas[gi, 2] = b

    return c_atlas


def save_color_atlas():
    onto = read_ontology("mouse_ontology.json")
    colors = id_colors(onto)
    atlas = np.load('contourify.npy')
    c_atlas = color_atlas(atlas, colors)
    np.save('color_atlas.npy', c_atlas)