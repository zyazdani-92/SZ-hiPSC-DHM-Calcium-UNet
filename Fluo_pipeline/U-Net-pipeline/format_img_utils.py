'''
Writer: Maxime Moreaud

'''

# General imports

# Imaging imports
import numpy as np

# Local imports


# -------------------- Gestion de format d'images -------------------- #

def images_reformat_from_list4D(images4D, typeOutputImage):
    """
    TB. images_reformat_from_list4D

    :param images4D: Images formatted as list4D (list(Height x Width x n))
    :param typeOutputImage: Image format desired in output
    :return: Returns input images (or the image) in the desired format

    divers_images_reformat will raise an error if conversion is impossible.
    Available formats :
        list2D : Invalid, transformed in list3D
        list3D : [Height x Width]. Discouraged, use listbatch3D instead
        listbatch3D : list(Height x Width)
        listbatch4D : list(batch-size x Height x Width)
        list4D : list(Height x Width x n)
        list5D : list(batch-size x Height x Width x n)
        2D : Height x Width
        3D : Height x Width x n
        batch3D : Batch-size x Height x Width
        4D : Batch-size x Height x Width x n
    """

    nbImages = len(images4D)

    # Output shaped as a list of images
    if typeOutputImage.startswith("list"):

        if (typeOutputImage.endswith("2D")) \
                or ((typeOutputImage.endswith("3D"))
                    and (not typeOutputImage.endswith("batch3D"))):
            if typeOutputImage.endswith("2D"):
                print("/!\\ Format d'output invalide. Utilisez list3D plutôt que list2D. Format transformé en list3D.")
                typeOutputImage = "list3D"

            if len(images4D) != 1:
                print("/!\\ Format d'output invalide. Pour plusieurs images, utilisez listbatch3D plutôt que "
                      + typeOutputImage + ". Format transformé en listbatch3D.")
                typeOutputImage = "listbatch3D"

        # list(Height x Width)
        if typeOutputImage.endswith("batch3D"):
            assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                              + ", impossible de convertir une image de " + str(images4D[0].shape[2]) \
                                              + " canaux vers une image de forme Height x Width"
            images = [img[:, :, 0] for img in images4D]

        # list(Batch-size x Height x Width)
        elif typeOutputImage.endswith("batch4D"):
            assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                              + ", impossible de convertir une image de " + str(images4D[0].shape[2]) \
                                              + " canaux vers une image de forme Height x Width"
            images = [img[:, :, 0][np.newaxis, ...] for img in images4D]

        # list(Height x Width x n)
        elif typeOutputImage.endswith("4D"):
            images = images4D

        # list(Batch-size x Height x Width x n)
        elif typeOutputImage.endswith("5D"):
            images = [img[np.newaxis, ...] for img in images4D]

        # [Height x Width]
        else:
            print("/!\\ list3D ([Height x Width]) est déconseillé --> listbatch3D (list(Height x Width))")
            assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                              + ", impossible de convertir une image de " + str(images4D[0].shape[2]) \
                                              + " canaux vers une image de forme Height x Width"
            images = [images4D[0][:, :, 0]]

    # Output shaped as a batch of images
    else:
        # Batch-size x Height x Width x n
        if typeOutputImage.endswith("4D"):
            images = [img[np.newaxis, ...] for img in images4D]
            images = np.concatenate(images, axis=0)

        # Height x Width
        elif typeOutputImage.endswith("2D"):
            assert len(images4D) == 1, "Reformatage impossible vers " + typeOutputImage + " depuis plusieurs images"
            assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                              + ", impossible de convertir une image de " + str(images4D[0].shape[2]) \
                                              + " canaux vers une image de forme Height x Width"
            images = images4D[0][:, :, 0]

        # Batch-size x Height x Width
        elif typeOutputImage.endswith("batch3D"):
            assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                              + ", impossible de convertir une image de " + str(images4D[0].shape[2]) \
                                              + " canaux vers une image de forme Height x Width"

            images = np.concatenate([images4D[i][:, :, 0][np.newaxis, ...] for i in range(nbImages)], axis=0)

        # Height x Width x n
        else:  # "3D"
            if len(images4D) > 1:
                assert images4D[0].shape[2] == 1, "Reformatage impossible vers " + typeOutputImage \
                                                  + " depuis plusieurs images ayant plusieurs canaux"
                images = np.concatenate(images4D, axis=2)
            else:
                images = images4D[0]

    return images


"""
# Test of images_reformat_from_list4D function
import numpy as np
from src.gestion_db_utils import format_img_utils as fi
a = [np.zeros([799,799, 1]) for i in range(1)]
b = [np.zeros([799,799, 5]) for i in range(1)]
c = [np.zeros([799,799, 1]) for i in range(3)]
d = [np.zeros([799,799, 5]) for i in range(3)]
e = [a, b, c, d, c, a, a, d, d, c, c]
print(fi.get_format_images(a) + "\n")
l = ["2D", "3D", "3D", "4D", "batch3D", "list2D", "list3D", "list4D", "list5D", "listbatch3D", "listbatch4D"]
for i in range(len(l)):
    print(fi.get_format_images(fi.images_reformat_from_list4D(e[i], l[i])))

# Should print: 
# list4D\n\n2D\n3D\n3D\n4D\nbatch3D\n/!\\ ...\n/!\\ ...\nlistbatch3D\n
# /!\\ ...\nlistbatch3D\nlist4D\nlist5D\nlistbatch3D\nlistbatch4D
"""


def images_reformat_to_list4D(images):
    """
    TB. images_reformat_to_list4D

    :param images: Images shaped freely
    :return: Returns images formatted as list4D (list(Height x Width x n))

    Check images_reformat_from_list4D to consult available formats
    """

    # Inputs check
    if isinstance(images, tuple):
        images = list(images)

    # List of images case
    if isinstance(images, list):
        imgShape = images[0].shape
        if len(imgShape) < 4:
            if len(imgShape) == 3:
                if not (abs(imgShape[2] - imgShape[1]) < abs(imgShape[0] - imgShape[1])):
                    return images
            else:
                images = [img[np.newaxis, ...] for img in images]
        images = np.concatenate(images, axis=0)

    imgShape = images.shape

    # No channel neither for grey levels, nor for batch-size
    if len(imgShape) == 2:
        images = [images[..., np.newaxis]]

    # Batch of images
    elif len(imgShape) == 4:
        images = [images[i, :, :, :] for i in range(images.shape[0])]

    # Channel for grey levels but in the first dimension instead of the third or batch of images without channel for
    # grey levels
    elif abs(imgShape[2] - imgShape[1]) < abs(imgShape[0] - imgShape[1]):
        images = [images[i, :, :][..., np.newaxis] for i in range(images.shape[0])]

    # Only one image with a channel for grey levels
    else:
        images = [images]

    return images


def get_format_images(images):
    """
    TB. get_format_images

    :param images: Images shaped freely
    :return: Returns input image format

    Check images_reformat_from_list4D to consult available formats
    """

    # Inputs check
    if isinstance(images, tuple):
        images = list(images)

    typeInputImage = ""

    # List of images case
    if isinstance(images, list):
        typeInputImage += "list"
        imgShape = images[0].shape
        if len(imgShape) < 4:
            if (len(imgShape) == 3) \
                    and (abs(imgShape[2] - imgShape[1]) < abs(imgShape[0] - imgShape[1])):
                return typeInputImage + "batch4D"
            images = [img[np.newaxis, ...] for img in images]
        else:
            return typeInputImage + "5D"
        # No need of concatenation because the output will be the same and in order to save some calcul duration.
        # Moreover, it could generate errors if images don't have the same size
        # images = np.concatenate(images, axis=0)
        images = images[0]

    imgShape = images.shape

    # No channel neither for grey levels, nor for batch-size
    if len(imgShape) == 2:
        typeInputImage += "2D"

    # Batch of images
    elif len(imgShape) == 4:
        typeInputImage += "4D"

    # Channel for grey levels but in the first dimension instead of the third or batch of images without channel for
    # grey levels
    elif abs(imgShape[2] - imgShape[1]) < abs(imgShape[0] - imgShape[1]):
        typeInputImage += "batch3D"

    # Only one image with a channel for grey levels
    else:
        typeInputImage += "3D"

    return typeInputImage


"""
# Test of get_format_images and images_reformat_to_list4D functions
import numpy as np
from src.gestion_db_utils import format_img_utils as fi
a = np.zeros([799,799])
b = np.repeat(a[..., np.newaxis], 2, axis=2)
c = np.repeat(a[np.newaxis, ...], 3, axis=0)
d = np.repeat(c[..., np.newaxis], 4, axis=3)
e = [a for i in range(5)]
f = [b for i in range(6)]
g = [c for i in range(7)]
h = [d for i in range(8)]
l = [a,b,c,d,e,f,g,h]
for ll in l:
    print(fi.get_format_images(ll))
    print(fi.get_format_images(fi.images_reformat_to_list4D(ll)) + "\n")

# Should print: 
# 2D\nlist4D\n\n3D\nlist4D\n\nbatch3D\nlist4D\n\n4D\nlist4D\n\n
# listbatch3D\nlist4D\n\nlist4D\nlist4D\n\nlistbatch4D\nlist4D\n\nlist5D\nlist4D
"""


def get_format_from_inputs(isList, batch_axis, nbChan):
    """
    TB. get_format_from_inputs

    :param isList: Are images shaped as a list
    :param batch_axis: Have images a channel (on first dimension) for batch of images
    :param nbChan: How many 2D images channels have they (0 for Width x Height, n for Width x Height x n)
    :return: Returns a format name from characteristics of images

    Check images_reformat_from_list4D to consult available formats
    """

    if isList:

        if nbChan == 0:
            if batch_axis:
                return "listbatch4D"
            else:
                return "listbatch3D"
        else:
            if batch_axis:
                return "list5D"
            else:
                return "list4D"

    else:
        if nbChan == 0:
            if batch_axis:
                return "batch3D"
            else:
                return "2D"
        else:
            if batch_axis:
                return "4D"
            else:
                return "3D"


"""
# Test of get_format_from_inputs function
import numpy as np
from src.gestion_db_utils import format_img_utils as fi
a = [True, False]
b = [True, False]
c = [0, 1, 2]
for aa in a:
    for bb in b:
        for cc in c:
            print(fi.get_format_from_inputs(aa, bb, cc))

# Doit print: 
# listbatch4D\nlist5D\nlist5D\nlistbatch3D\nlist4D\nlist4D\nbatch3D\n4D\n4D\n2D\n3D\n3D
"""


def images_reformat(images, typeOutputImage):
    """
    TB. images_reformat

    :param images: Images shaped freely
    :param typeOutputImage: Output images format desired
    :return: Retourne les images (l'image) d'entrées sous le format souhaité

    Check images_reformat_from_list4D to consult available formats
    """

    typeInputImage = get_format_images(images)

    if typeInputImage != typeOutputImage:
        return images_reformat_from_list4D(images_reformat_to_list4D(images), typeOutputImage)

    else:
        return images
