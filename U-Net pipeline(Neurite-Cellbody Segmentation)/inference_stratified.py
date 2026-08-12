# General imports
import time
from datetime import datetime

# Imaging imports
import numpy as np
import cv2

# Local imports
#from src import format_img_utils
from format_img_utils import *

# -------------------- Main function for stochastic inference -------------------- #

def inference_fullimage_stratified(
        model, _IM, szPatch=(192, 192), nbChanOut=1, nbpassstratified=1, prediction_fct=None,
        input_list_instead_chan=False, input_bach_axis=False, input_nbChan=None, output_checkShape=True,
        shufflePatchs=False,
        normInput=False, normPatch=False, normPatchRobust=False, pctNormRobust=0.005, thresh01normRob=True,
        normPatchOutFromIn=False, normPatchOutFromInRobust=False, normOutFromIn=False, normOutFromInRobust=False,
        patchModePreviewInterval=0, showPreview=False, showDuration=False):
    """
    inference_fullimage_stratified
    This function comes from a C# program gave by Maxime Moreaud

    :param model: Trained model
    :param _IM: Input image of the network shaped as np.array([H, W, nbChan]) in np.float32
    :param szPatch: Size of input patches of the model
    :param nbChanOut: Number of channels of a input image
    :param nbpassstratified: Number of pass on the whole image by random patches
    :param prediction_fct: Prediction function, None for model.predict
    :param input_list_instead_chan: Boolean, input of prediction_fct is a list of images if true
    :param input_bach_axis: Boolean, input image(s) of prediction_fct have an axis for batch-size if true
    :param input_nbChan: Boolean, input image(s) of prediction_fct have an axis channels (grey levels) if true
    :param output_checkShape: inference_fullimage_stratified have to check output of prediction_fct shape, elsewhere it
                will raise an error if it's different from Height x Width x n. That permits to reduce a little bit
                calcul duration
    :param shufflePatchs: Shuffle patches inference order during a pass, that doesn't change the output but it can give
                a better visualisation of randomness (showPreview)
    :param normInput: Norm, ou not, input _IM
    :param normPatch: Norm, ou not, a patch before prediction of prediction_fct
    :param normPatchRobust: Norm, ou not, a patch before prediction of prediction_fct following quantiles of its
                histogram
    :param pctNormRobust: Rate used for robust normalizations
    :param thresh01normRob: Threshold, or not, to 0/1 after robust normalizations
    :param normPatchOutFromIn: Apply, or not, quantitativity of an input patch on the output one
    :param normPatchOutFromInRobust: Apply, or not, quantitativity of an input patch on the output one with quantiles
    :param normOutFromIn: Apply, or not, quantitativity of input image on the output one
    :param normOutFromInRobust: Apply, or not, quantitativity of input image on the output one with quantiles
    :param patchModePreviewInterval: Interval of feedback (prompt and preview), 0 for no feed-back
    :param showPreview: Displays, or not, current output and weights applied every patchModePreviewInterval
    :param showDuration: Displays, or not, duration in seconds of the inference of the whole image
    :return: Returns output image for the trained network shapes as np.array([H, W, nbChan]) in np.float32 by stochastic
                patching method

    Rq : To add an axis at the end of an image: _IM = _IM[..., np.newaxis]. To delete it: _IM = _IM[:, :, 0] if there
                is only one channel

    This inference method is the stochastic one. It apply a random patching on the whole image, infer every patch, keep
    only 2/3 of the center of the output patch, and makes a weighted sum of all predicted patches giving a more
    important weight for the center of them.
    """

    start_time = time.time() if showDuration else None
    np.random.seed(int(datetime.now().strftime("%Y%m%d%H%M%S")) % 2**31)

    # Initialisation output image
    imSz = _IM.shape
    _H = imSz[0]
    _W = imSz[1]
    nbChanIn = imSz[2]
    if input_nbChan is None:
        input_nbChan = nbChanIn
    input_format = get_format_from_inputs(input_list_instead_chan, input_bach_axis, input_nbChan)
    _OUT = np.zeros([_H, _W, nbChanOut], dtype=np.float32)

    # Initialisation of patches
    H = szPatch[0]
    W = szPatch[1]
    IM = np.zeros([H, W, nbChanIn], dtype=np.float32)
    OUT = np.zeros([H, W, nbChanOut], dtype=np.float32)

    # Input min/max
    imin, imax = findMinMax(_IM, nbChan=nbChanIn)

    # Normalization of input
    if normInput:
        NormPatch(_IM, imin, imax)

    # Counts the number of patches need for each iteration
    nbtotalpatch = 0
    for y in range((-5 * H) // 6 + 1, _H - (H // 6) - (H // 3) + 1, H // 3):
        for x in range((-5 * W) // 6 + 1, _W - (W // 6) - (W // 3) + 1, W // 3):
            nbtotalpatch += 1

    # Random position of patches
    PY = np.zeros([nbtotalpatch], dtype=int)
    PX = np.zeros([nbtotalpatch], dtype=int)

    u = 0
    for y in range((-5 * H) // 6 + 1, _H - (H // 6) - (H // 3) + 1, H // 3):
        for x in range((-5 * W) // 6 + 1, _W - (W // 6) - (W // 3) + 1, W // 3):
            aleay = np.random.randint(0, W / 3)
            aleax = np.random.randint(0, H / 3)

            PY[u] = y + aleay
            PX[u] = x + aleax
            u += 1

    # Shuffle of the ordre of inference of patches
    if shufflePatchs:
        Shuffle(PX, PY)

    # Weighted map of a patch creation
    POND = np.zeros([H, W, 1], dtype=np.float32)
    for j in range(H):
        for i in range(W):
            POND[j, i, :] = (j - H // 2) * (j - H // 2) + (i - W // 2) * (i - W // 2)
            POND[j, i, :] = ((W * W) - POND[j, i, :]) / (W * W)
            if POND[j, i, :] < 0:
                POND[j, i, :] = 0

    # Weighted map for the final output
    _COUNT = np.zeros([_H, _W, 1], dtype=np.float32)

    # Loop for the number of iterations of inference by patches if the whole image
    for a in range(nbpassstratified):
        c = 0

        # Loop on patches for image inference
        for u in range(nbtotalpatch):
            _y = PY[u]
            _x = PX[u]

            # Creation of an input patches to infer it, mirrored image for border management
            for j in range(H):
                jj = j + _y
                if jj >= _H:
                    jj = 2 * _H - jj - 1
                if jj < 0:
                    jj = -jj

                for i in range(W):
                    ii = i + _x
                    if ii >= _W:
                        ii = 2 * _W - ii - 1
                    if ii < 0:
                        ii = -ii

                    IM[j, i, :] = _IM[jj, ii, :]

            # min/max of input patch
            inmin, inmax = findMinMax(IM, nbChan=nbChanIn)
            # Normalization of the patch individually
            if normPatch:
                if normPatchRobust:
                    NormPatchRobust(nbChanIn, IM, inmin, inmax, pourc=pctNormRobust, thresh01=thresh01normRob)
                else:
                    NormPatch(IM, inmin, inmax)

            # Prediction of the output patch by the trained model
            # Output is reshaped as Height x Width x n
            if output_checkShape:
                if prediction_fct is None:
                    outPatch = model.predict(images_reformat(IM, input_format))
                else:
                    outPatch = prediction_fct(model, images_reformat(IM, input_format))
                OUT[:, :, :] = images_reformat(outPatch, "3D")

            # Output has the shape Height x Width x n
            else:
                if prediction_fct is None:
                    OUT[:, :, :] = model.predict(images_reformat(IM, input_format))
                else:
                    OUT[:, :, :] = prediction_fct(model, images_reformat(IM, input_format))

            # Un-normalization of output patch to get input quantitativity
            if normPatchOutFromIn:
                oumin, oumax = findMinMax(OUT, nbChan=nbChanOut)
                new_oumin, new_oumax = inmin, inmax

                if normPatchOutFromInRobust:
                    NormPatchRobust(nbChanOut, OUT, oumin, oumax, pourc=pctNormRobust, thresh01=thresh01normRob)
                    new_oumin, new_oumax = findMinMaxRobust(IM, nbChan=nbChanIn, pourc=pctNormRobust)
                else:
                    NormPatch(OUT, oumin, oumax)

                StandPatch(OUT, new_oumin, new_oumax)

            # Weighting of output patch and calculation of final weighting. 1/6 of borders are deleted and management of
            # patches exceeding the whole image
            jjmin = max(0, H//6+_y)
            jjmax = min(_H, H-(H//6)+_y)
            jpmin = H//6 + jjmin - (H//6+_y)
            jpmax = H-(H//6) - ((H-(H//6)+_y) - jjmax)

            iimin = max(0, W//6+_x)
            iimax = min(_W, W-(W//6)+_x)
            ipmin = W//6 + iimin - (W//6+_x)
            ipmax = W-(W//6) + iimax - (W-(W//6)+_x)

            _OUT[jjmin:jjmax, iimin:iimax, :] += \
                POND[jpmin:jpmax, ipmin:ipmax, :] * OUT[jpmin:jpmax, ipmin:ipmax, :]
            _COUNT[jjmin:jjmax, iimin:iimax, :] += POND[jpmin:jpmax, ipmin:ipmax, :]

            # Displaying in terminal the current step
            if patchModePreviewInterval != 0:
                if c == 0:
                    print("Pass " + str(a) + " of " + str(nbpassstratified))
                c += 1
                if c % patchModePreviewInterval == 0:
                    print("Pass " + str(a) + " of " + str(nbpassstratified)
                          + " process patch " + str(c) + " of " + str(nbtotalpatch))

                    if showPreview:
                        min4disp, max4disp = findMinMax(_OUT, nbChan=nbChanOut)
                        dispOut = \
                            [(_OUT[:, :, z] - min4disp[z]) / (max4disp[z] - min4disp[z]) for z in range(nbChanOut)]
                        min4disp, max4disp = findMinMax(_COUNT, nbChan=1)

                        cv2.namedWindow("Current Output (left) and Current Count (right)", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Current Output (left) and Current Count (right)", 1600, 1600//(nbChanOut+1))
                        cv2.imshow("Current Output (left) and Current Count (right)",
                                   cv2.hconcat(dispOut
                                               + [(_COUNT[:, :, 0] - min4disp[0]) / (max4disp[0] - min4disp[0])]))
                        cv2.waitKey(0)

    # Normalization regarding the number of patches used of each pixel
    _OUT /= _COUNT

    # Output data finalization
    omin, omax = findMinMax(_OUT, nbChan=nbChanOut)

    # Un-normalization of the output to get input quantitavity
    if normOutFromIn:
        new_omin, new_omax = imin, imax

        if normOutFromInRobust:
            NormPatchRobust(nbChanOut, _OUT, omin, omax, pourc=pctNormRobust, thresh01=thresh01normRob)
            new_omin, new_omax = findMinMaxRobust(_IM, nbChan=nbChanIn, pourc=pctNormRobust)
        else:
            NormPatch(_OUT, omin, omax)

        StandPatch(_OUT, new_omin, new_omax)

    if showDuration:
        print("Durée de l'inférence : %s secondes ---" % (time.time() - start_time))

    return _OUT


# -------------------- Utils functions for stochastic inference : normalization -------------------- #

def findMinMax(IN, nbChan=None):
    """
    findMinMax
    This function comes from a C# program gave by Maxime Moreaud

    :param IN: Image np.array [Height x Width x nbChan] in np.float32
    :param nbChan: Number of channels of input image, automatically calculated if None
    :return: Returns two np.array in np.float32 giving respectively min and max of each channel of the input image
    """

    # Inputs check
    if len(IN.shape) == 2:
        IN = IN[..., np.newaxis]
    if nbChan is None:
        nbChan = IN.shape[2]

    inmin = np.zeros([nbChan], dtype=np.float32)
    inmax = np.zeros([nbChan], dtype=np.float32)

    for z in range(nbChan):
        inmin[z] = np.min(IN[:, :, z])
        inmax[z] = np.max(IN[:, :, z])
        if inmax[z] - inmin[z] <= 0:
            print("/!\\ Dynamic error.")
            return False

    return inmin, inmax


def findMinMaxRobust(IN, nbChan=None, pourc=0.005):
    """
    findMinMaxRobust
    This function comes from a C# program gave by Maxime Moreaud

    :param IN: Image np.array [Height x Width x nbChan] in np.float32
    :param nbChan: Number of channels of input image, automatically calculated if None
    :param pourc: Rate of quantiles researched
    :return: Returns two np.array in np.float32 giving respectively min and max quantiles of each channel of the input
                image
    """

    # Inputs check
    if len(IN.shape) == 2:
        IN = IN[..., np.newaxis]
    if nbChan is None:
        nbChan = IN.shape[2]

    inmin = np.zeros([nbChan], dtype=np.float32)
    inmax = np.zeros([nbChan], dtype=np.float32)

    for z in range(nbChan):
        inmin[z] = np.quantile(IN[:, :, z], pourc)
        inmax[z] = np.quantile(IN[:, :, z], 1-pourc)
        if inmax[z] - inmin[z] <= 0:
            print("/!\\ Dynamic error.")
            return False

    return inmin, inmax


def NormPatch(IN, inmin, inmax):
    """
    NormPatch
    This function comes from a C# program gave by Maxime Moreaud

    :param IN: Image np.array [Height x Width x nbChan] in np.float32
    :param inmin: np.array in np.float32 giving mins of each channel of IN
    :param inmax: np.array in np.float32 giving maxs of each channel of IN
    :return: Transforms input image to normalize it (between 0 and 1). Returns True when finished
    """

    IN[:, :, :] = (IN[:, :, :] - inmin) / (inmax - inmin)
    return True


def NormPatchRobust(nbChan, IN, inminZ, inmaxZ, pourc=0.005, thresh01=True):
    """
    NormPatchRobust
    This function comes from a C# program gave by Maxime Moreaud

    :param nbChan: Number of channels of input image, automatically calculated if None
    :param IN: Image np.array [Height x Width x nbChan] in np.float32
    :param inminZ: np.array in np.float32 giving mins of each channel of IN
    :param inmaxZ: np.array in np.float32 giving maxs of each channel of IN
    :param pourc: Rate of threshold values of output image
    :param thresh01: Threshold to 0 and 1 of the output
    :return: Transforms input image to normalize it (between 0 and 1) in a more robust way by thresholding histogram of
                the image within a rate of values (quantiles). Returns True when finished
    """

    # Inputs check
    if nbChan is None:
        nbChan = IN.shape[2]

    if (inminZ is None) or (inmaxZ is None):
        inmin, inmax = findMinMax(IN, nbChan=nbChan)
        if inminZ is None:
            inminZ = inmin
        if inmaxZ is None:
            inmaxZ = inmax

    for z in range(nbChan):
        inmin = inminZ[z]
        inmax = inmaxZ[z]

        cste = 0

        if inmin != inmax:
            cste = 1 / (inmax - inmin)

        # Seeking for quantiles
        if (pourc > 0) and (inmax != inmin):
            VMAXold = inmax
            VMINold = inmin

            # Min/Max quantiles
            inmin = np.quantile(IN[:, :, z], pourc)
            inmax = np.quantile(IN[:, :, z], 1-pourc)

            if inmin == inmax:
                inmax += VMAXold - VMINold

            cste = 1 / (inmax - inmin)

        # Un-normalization on new values
        IN[:, :, z] = (cste * (IN[:, :, z] - inmin))

        # Threshold at 0 and 1
        if thresh01:
            IN[:, :, z] = np.maximum(IN[:, :, z], 0)
            IN[:, :, z] = np.minimum(IN[:, :, z], 1)

        inminZ[z], inmaxZ[z] = findMinMax(IN, nbChan)

    return True


def StandPatch(IN, inmin, inmax):
    """
    StandPatch
    This function comes from a C# program gave by Maxime Moreaud

    :param IN: Image np.array [Height x Width x nbChan] in np.float32
    :param inmin: np.array in np.float32 giving mins of each channel of IN
    :param inmax: np.array in np.float32 giving maxs of each channel of IN
    :return: Transfoms the input normalized image in order to set its extremas. Returns True when finished
    """

    IN[:, :, :] = IN[:, :, :] * (inmax - inmin) + inmin
    return True


# -------------------- Utils functions for stochastic inference : shuffle of patches -------------------- #

def Shuffle(PX, PY):
    """
    Shuffle
    This function comes from a C# program gave by Maxime Moreaud

    :param PX: List of numbers (X positions in an image)
    :param PY: List of numbers (Y positions in an image)
    :return: Transforms input lists by making a permutation of themselvesmutation aléatoire. Returns True when finished
    """

    idx = np.random.permutation(np.arange(PX.shape[0]))
    PX[:] = PX[idx]
    PY[:] = PY[idx]
    return True
