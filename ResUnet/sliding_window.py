import math

def generateSlices(im_shape, maxWindowSize, overlapPercent):
    """
    Generates a set of sliding windows for a dataset with the specified dimensions and order.
    """

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    windowSizeX = min(maxWindowSize[2], im_shape[2])
    windowSizeY = min(maxWindowSize[1], im_shape[1])
    windowSizeZ = min(maxWindowSize[0], im_shape[0])

    # Compute the window overlap and step size
    windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
    windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
    windowOverlapZ = int(math.floor(windowSizeZ * overlapPercent))
    stepSizeX = windowSizeX - windowOverlapX
    stepSizeY = windowSizeY - windowOverlapY
    stepSizeZ = windowSizeZ - windowOverlapZ

    # Determine how many windows we will need in order to cover the input data
    lastX = im_shape[2] - windowSizeX
    lastY = im_shape[1] - windowSizeY
    lastZ = im_shape[0] - windowSizeZ
    xOffsets = list(range(0, lastX+1, stepSizeX))
    yOffsets = list(range(0, lastY+1, stepSizeY))
    zOffsets = list(range(0, lastZ+1, stepSizeZ))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(xOffsets) == 0 or xOffsets[-1] != lastX:
        xOffsets.append(lastX)
    if len(yOffsets) == 0 or yOffsets[-1] != lastY:
        yOffsets.append(lastY)
    if len(zOffsets) == 0 or zOffsets[-1] != lastZ:
        zOffsets.append(lastZ)

    # Generate the list of windows
    windows = []
    for xOffset in xOffsets:
        for yOffset in yOffsets:
            for zOffset in zOffsets:
                windows.append((
                    slice(zOffset, zOffset+windowSizeZ),
                    slice(yOffset, yOffset+windowSizeY),
                    slice(xOffset, xOffset+windowSizeX)
                ))
    overlap = (windowOverlapZ, windowOverlapY, windowOverlapX)
    return windows, overlap

def generateSlices2D(im_shape, maxWindowSize, overlapPercent):
    """
    Generates a set of sliding windows for a dataset with the specified dimensions and order.
    """

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    windowSizeX = min(maxWindowSize[1], im_shape[1])
    windowSizeY = min(maxWindowSize[0], im_shape[0])

    # Compute the window overlap and step size
    windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
    windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
    stepSizeX = windowSizeX - windowOverlapX
    stepSizeY = windowSizeY - windowOverlapY

    # Determine how many windows we will need in order to cover the input data
    lastX = im_shape[1] - windowSizeX
    lastY = im_shape[0] - windowSizeY
    xOffsets = list(range(0, lastX+1, stepSizeX))
    yOffsets = list(range(0, lastY+1, stepSizeY))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(xOffsets) == 0 or xOffsets[-1] != lastX:
        xOffsets.append(lastX)
    if len(yOffsets) == 0 or yOffsets[-1] != lastY:
        yOffsets.append(lastY)

    # Generate the list of windows
    windows = []
    for xOffset in xOffsets:
        for yOffset in yOffsets:
            windows.append((
                slice(yOffset, yOffset+windowSizeY),
                slice(xOffset, xOffset+windowSizeX)
            ))
    overlap = (windowOverlapY, windowOverlapX)
    return windows, overlap