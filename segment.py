"""
Image segmentation using edge detection.
Author: Leonidas Papadakis

This program takes an image file as input and segments it into foreground 
and background regions using a custom Canny edge detection pipeline. 
Uses morphological closing and flood fill on formed edges to separate foreground from background.
It is designed to work well on images of natural scenes with complex lighting and shadows.

Segmentation pipeline:
1 - High-pass filtering in frequency domain to enhance edges and reduce lighting gradients
2 - Sobel gradient computation for edge magnitude and direction
3 - Non-maximum suppression to thin edges to one pixel thick
4 - Hysteresis thresholding to connect weak edges to strong edges
5 - Morphological closing to fill gaps in edges
6 - Flood fill from top of image border to separate background from foreground

Comparison to an alternative segmentation method, K-means clustering:
    Advantages:
    - Better handles natural scenes with complex lighting and shadows
    - More robust to varying object colors and textures
    - Produces cleaner boundaries around objects
    Disadvantages:
    - More computationally intensive
    - Requires careful parameter tuning for different scenes
    - May fail if object borders are very weak or blurred
    - Sensitive to noise in highly textured regions

Another alternative approach is using Otsu threshold with a magnitude histogram,
which is more automatic, however it assumes bimodal distribution, which is rare 
in natural images.

The frequency-domain preprocessing addresses the main weakness of edge-based
methods by suppressing illumination gradients and enhancing real object borders.

Usage:
    python segment.py -i input.jpg -o mask.png -v visual.png
    python segment.py -i input.jpg -o mask.png -v visual.png -c 0.1 s 2.0 -l 75 -u 85 -k 5

Help: python segment.py -h

Dependencies:
    numpy: Array processing and FFT operations
    scipy: Convolution operations
    PIL: Image I/O
"""

import sys, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from collections import deque

def loadImage(fileName: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from a file and converts to numpy array.
    Returns both RGB colour and grayscale versions.
    The grayscale version is used for edge detection, whilest the colour version
    is preserved for creating the visual output.
    
    Parameters:
        fileName : str
            Path to the input image file. 
            Supports formats supported by PIL (JPG, PNG, BMP, etc.)
    Returns:
        tuple[np.ndarray, np.ndarray] containing:
            colour : np.ndarray
                3D numpy array of RGB pixel values.
            grayscale : np.ndarray
                2D numpy array of grayscale pixel values
                in range [0, 255]
    """
    try:
        # Read coloured and greyscale image
        image = Image.open(fileName)
        colour = np.array(image.convert("RGB"))
        greyscale = np.array(image.convert("L"))

    except Exception as e:
        sys.exit(e)

    return colour, greyscale

def saveImage(fileName: str, image: np.ndarray) -> None:
    """
    Converts a numpy array to an image and saves it to an image file.
    
    Parameters:
        fileName : str
            Path where the output image file will be saved.
        image : np.ndarray
            Numpy array of pixel values in range [0, 255].    
    """

    try:
        Image.fromarray(image).save(fileName)
        print(f"Image saved to {fileName}")
    except Exception as e:
        sys.exit(f"Error writing {fileName}: {e}")

def createHighpassFilter(shape : tuple, cutoff, sigma) -> np.ndarray:
    """
    This function creates a Gaussian high pass filter to suppress low frequencies (such as 
    smooth regions and light gradients) while preserving high frequencies (such as edges 
    and fine details).
    Method chosen over Butterworth or ideal high-pass due to smoothness (no ringing).
    Works well for removing lighting gradients and enhancing object borders.
    Aggressive cutoffs may amplify noise.

    Arguments:
        shape : tuple[int, int]
            Shape of the input image (height, width).
        cutoff : float, default = 0.05
            Cutoff frequency as a fraction.
            Controls how aggressively low frequencies are suppressed.
            Smaller values give a stronger high-pass while larger values give a gentler 
            high-pass.
        sigma : float, default = 4.0
            Scaling factor for standard deviation.
            Larger values produce smoother transitions whereas smaller values give sharper
            transitions.
    Returns:
        np.ndarray
            filter mask in the frequency domain
    """
    
    rows, columns = shape
    centreRow, centreColumn = rows // 2, columns // 2

    # Create coordinate grid for distance matrix
    y, x = np.ogrid[-centreRow:rows-centreRow, 
                    -centreColumn:columns-centreColumn]
    
    # Create distance matrix using pythagorean theorem
    distance = np.sqrt(x**2 + y**2)
    
    # Create Gaussian high-pass filter
    filter = 1 - np.exp( -(distance**2) / 
                     (2 * (cutoff * min(rows, columns) * sigma)**2))
    
    return filter


def enhance(input : np.ndarray, cutoff, sigma) -> np.ndarray:
    """
    Enhance a grayscale image edges by applying a Gaussian high-pass filter in
    the frequency domain. 
    Preprocesses the image to suppress low-frequency areas (lighting,
    shadows) and enhance high-frequency areas (edges, textures) before
    edge detection.
    Chosen frequency-domain approach over spatial-domain because it more precisely 
    handles global gradients (common in natural outdoor scenes). Spatial-domain 
    filtering is simpler but can introduce halo artifacts.
    Greatly improves edge detection in low-contrast regions. However, may amplify 
    high-frequency noise in detailed areas (e.g. grass, leaves).

    Parameters:
        input : np.ndarray
            Grayscale image as a 2D numpy array.
        cutoff : float, default = 0.07
            Cutoff frequency for the high-pass filter.
    Returns:
        np.ndarray
            Enhanced image (float64, normalized to [0, 255] range).
    """

    # Convert to frequency domain
    frequencies = np.fft.fft2(input.astype(np.float64))

    # Shift to centre
    frequencies = np.fft.fftshift(frequencies)
    
    # Create and apply high-pass filter
    frequencies *= createHighpassFilter(input.shape, cutoff, sigma)
    
    # Inverse shift and convert back to space domain
    frequencies = np.fft.ifftshift(frequencies)
    enhanced = np.abs(np.fft.ifft2(frequencies))

    # Normalise to 0 - 255
    if (enhanced.max() - enhanced.min()) != 0:
        enhanced = 255 * (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    
    return enhanced.astype(np.float64)


def sobelGradients(image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function computes gradient magnitude and direction of each pixel 
    using 5x5 Sobel kernels.

    Sobel chosen for its robustness to noise compared to simpler methods.
    Performs well on natural images after preprocessing, but is sensitive to
    noise in fine textures. 

    5x5 kernels provide better noise robustness than 3x3 kernels.
    Direction is calculated in range [0-180] (edges are bidirectional) for use in 
    non-maxima suppression.

    This method assumes gradients are well defined at object edges. May fail with
    very fine textures or faint object borders.
    
    Parameters:
        image : np.ndarray
            Grayscale image as a 2D numpy array.
    Returns a tuple containing:
        magnitude : np.ndarray
            Gradient magnitude of each pixel.
        direction : np.ndarray
            Gradient direction in degrees (0, 180), used for non-maxima 
            suppression.
    """
    
    # Create sobel kernel for horizontal and vertical gradients
    xKernal = np.array([[2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2], 
                        [4, 2, 0, -2, -4], 
                        [2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2]])
    yKernal = np.array([[ 2,  2,  4,  2,  2], 
                        [ 1,  1,  2,  1,  1],
                        [ 0,  0,  0,  0,  0],
                        [-1, -1, -2, -1, -1],
                        [-2, -2, -4, -2, -2]])
    # Convolve with kernels in both directions to get x and y gradients
    xGradient = convolve(image, xKernal)
    yGradient = convolve(image, yKernal)
    
    # Calculate gradient magnitude using Pythagorean theorem
    magnitude = np.sqrt(xGradient ** 2 + yGradient ** 2).astype(np.uint8)

    # Calculate gradient direction in degrees, used for non-maxima suppression
    direction = np.arctan2(yGradient, xGradient) * (180 / np.pi) % 180

    return magnitude, direction

def findEdges(magnitude : np.ndarray, weakThreshold : int, StrongThreshold : int) -> np.ndarray:
    """
    This function uses hysteresis thresholding to find strong edges and weak edges, 
    then keeps only weak edges that are connected to strong edges using a breadth-first 
    search.
    Percentile thresholds chosen over fixed values for supporting varying contrast levels 
    in natural images.
    This method considerably reduces edge fragmentation compared to single thresholding.

    Parameters:
        magnitude : np.ndarray
            Gradient magnitude of each pixel as a 2D numpy array.
        weakThreshold : int
            Percentile for weak edge threshold.
        strongThreshold : int
            Percentile for strong edge threshold.

    Returns:
        np.ndarray[bool]
            Boolean mask of edges, where 0 = no edge, 1 = edge
    """
    
    # Find strong and weak edges
    weakPercentile = np.percentile(magnitude, weakThreshold)
    strongPercentile = np.percentile(magnitude, StrongThreshold)
    weakEdges = magnitude >= weakPercentile
    strongEdges = magnitude >= strongPercentile
    
    # Dimensions of image
    height, width = magnitude.shape

    # Use BFS to find strong edges connected to weak edges
    queue = deque(np.argwhere(strongEdges))

    # Relative positions of neighbouring pixels
    neighbours = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1)]
    while queue:

        # Pop strong edge from queue
        strongX, strongY = queue.popleft()
        
        # Iterate over neighbours
        for differenceX, differenceY in neighbours:
            neighborX, neighborY = strongX + differenceX, strongY + differenceY

            # Check neighbour is within bounds and is a weak edge
            if 0 <= neighborX < height and 0 <= neighborY < width \
                    and weakEdges[neighborX, neighborY]:
                
                # Promote neighbour to strong edge
                strongEdges[neighborX, neighborY] = True
                weakEdges[neighborX, neighborY] = False

                # Add neighbour to queue
                queue.append((neighborX, neighborY))

    return strongEdges


def nonMaximaSuppression(magnitude : np.ndarray, direction : np.ndarray) -> np.ndarray:
    """
    This function applies the non-maximum suppression step of the Canny edge detector 
    to thin edges to one pixel wide.
    For each pixel, the gradient magnitude is compared along the direction of the 
    gradient. If the current pixel's magnitude is not the maximum in that direction,
    it is set to zero.
    Gradient direction is seperated into four bins (horizontal, vertical, and two diagonals)
    for efficient comparison.
    Sensitive to innacurate direction estimates due to noise causing incorrect suppression.

    Parameters:
        magnitude : np.ndarray
            2D array of gradient magnitudes for each pixel.
        direction : np.ndarray
            2D array of gradient directions for each pixel in degrees [0-180]
    Returns:
        np.ndarray
            Suppressed map of magnitudes.
            Non-maximum pixels are set to zero, local maximums retain their original values.
    """

    # Image dimensions
    height, width = magnitude.shape

    # Output array – will keep only local maxima
    edgeMask = np.zeros_like(magnitude)
    
    # Loop over image, ignoring border pixels
    for x in range(1, height - 1):
        for y in range(1, width - 1):

            # Current pixel's gradient direction
            angle = direction[x, y]

            # Compare neighbors along gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180): # Horizontal gradient
                neighbors = [magnitude[x, y-1], magnitude[x, y+1]]

            # Diagonal NW – SE
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[x-1, y-1], magnitude[x+1, y+1]]

            # Vertical
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[x-1, y], magnitude[x+1, y]]

            # Diagonal SW – NE
            else:
                neighbors = [magnitude[x-1, y+1], magnitude[x+1, y-1]]

            # Keep value only if it is >= both neighbors along gradient
            if magnitude[x, y] >= max(neighbors):
                edgeMask[x, y] = magnitude[x, y]

    return edgeMask

def closeEdges(edges : np.ndarray, kernelSize : int) -> np.ndarray:
    """
    The most common failure of an edge based approach in natural scenes is 
    due to leakage from open edges when flood-filling. This function solves this 
    problem by closing small gaps in the edge mask.
    This function morphologicaly closes broken edges in a binary mask of edges.
    It dilates then erodes the binary edge mask to bridge small gaps caused by
    weak or low contrast boundaries.

    Parameters:
        edges : np.ndarray[bool]
            Binary edge map.
        kernelSize : int, default = 3
            Size of kernel used for dilation and erosion, forced to be odd.
            Larger values close bigger gaps but will thicken edges and risk losing
            distinct boundaries.
    Returns:
        np.ndarray[bool]
            Binary mask of closed edges, where 0 = no edge, 1 = edge.
    """

    # Ensure kernel size is >= 3 and odd
    if kernelSize < 3:
        kernelSize = 3
    elif kernelSize % 2 == 0:
        kernelSize += 1
    
    # Pad image with half of kernel size
    pad = kernelSize // 2
    padded = np.pad(edges > 0, pad, mode='constant', constant_values=False)

    # Loop through image and take max over neighborhood - dilation
    dilated = np.zeros_like(padded)
    for x in range(pad, padded.shape[0]-pad):
        for y in range(pad, padded.shape[1]-pad):
            dilated[x, y] = np.max(padded[x - pad : x + pad + 1, 
                                            y - pad : y + pad + 1])

    # Loop through image and take min over neighborhood - erosion
    erroded = np.zeros_like(dilated)
    for x in range(pad, dilated.shape[0]-pad):
        for y in range(pad, dilated.shape[1]-pad):
            erroded[x, y] = np.min(dilated[x - pad : x + pad + 1, 
                                          y - pad : y + pad + 1])

    # Crop back to original size
    return erroded[pad:-pad, pad:-pad]

def floodFillBackground(edges : np.ndarray) -> np.ndarray:
    """
    Seperate foreground by flooding background from the top image border.
    Assumes background is connected to the top border, which is a
    common case for natural scenes (e.g. landscapes, objects in nature, etc).
    Fails if forground has weak or low contrast boundaries, but this is
    mitigated by prior closeEdges().

    Parameters:
        edges : np.ndarray (bool)
            Binary mask of edges (0 = no edge, 1 = edge).
    Returns:
        np.ndarray (bool)
            Foreground mask (0 = background, 1 = foreground).
    """
    
    # Dimensions of image
    height, width = edges.shape

    # Create output foreground mask
    foreground = np.ones((height, width), bool)

    # Add all top border pixels that are not edges to a queue
    queue = deque()
    for column in range(width):   
        if not edges[0, column]: 
            queue.append((0, column))
            foreground[0, column] = False
        
    # Four relative positions for each direction: left, right, up, down
    directions = [(1,0), (-1,0), (0,-1), (0,1)]

    # Iterate over queue and flood fill
    while queue:
        backgroundRow, backgroundColumn = queue.popleft()

        # Search in all directions
        for differenceRow, differenceCollumn in directions:
            neighborRow, neighborCollumn = backgroundRow + differenceRow, backgroundColumn + differenceCollumn

            # Check neighbor is within bounds and is currently foreground
            if 0 <= neighborRow < height and 0 <= neighborCollumn < width \
                    and foreground[neighborRow, neighborCollumn]:
                
                # Mark neighbor as background
                foreground[neighborRow, neighborCollumn] = False

                # Add neighbor to queue if not an edge
                if not edges[neighborRow, neighborCollumn]:
                    queue.append((neighborRow, neighborCollumn))

    return foreground

def main(args):

    # Read input image and convert to grayscale
    colour, greyscale = loadImage(args.input_path)

    # Dimensions of image
    imageHeight, imageWidth = greyscale.shape

    # Apply FFT high-pass filter
    print("Enhancing image...")
    enhanced = enhance(greyscale, args.enhance_cutoff, args.enhance_sigma)

    # Compute gradient magnitude and direction for edge detection
    print("Computing Sobel gradients...")
    magnitude, direction = sobelGradients(enhanced)
    
    # Suppress gradients to 1 pixel thick
    print("Suppressing non-maxima gradients...")
    suppressedEdges = nonMaximaSuppression(magnitude, direction)

    # Connect strong edges to weak edges
    print("Connecting strong edges to weak edges...")
    edges = findEdges(suppressedEdges, args.lower_threshold, args.upper_threshold)

    # Close gaps in edges
    print("Closing gaps in edges...")
    edges = closeEdges(edges, kernelSize = args.closing_kernel_size)
    
    # Crop image border to prevent leakage when flood filling (edge detection is innacurate at borders)
    borderSize = min(imageHeight, imageWidth) // 700
    croppedEdges = edges[borderSize : imageHeight - borderSize, 
                          borderSize : imageWidth - borderSize]

    # Flood fill background from image border
    print("Flood filling background...")
    mask = floodFillBackground(croppedEdges)

    # Pad mask to original size
    mask = np.pad(mask, borderSize, mode='reflect')
    print("Segmentation complete.")

    # Save mask as an image file
    saveImage(args.output_path, mask.astype(np.uint8) * 255)

    # Convert mask and background to 3-channel if input has 3 dimension
    if np.ndim(colour) == 3:
        mask = np.repeat(mask[..., None], 3, axis=2)
        greyscale = np.repeat(greyscale[..., None], 3, axis=2)

        # Add alpha channel if input is RGBA
        if colour.shape[2] == 4:
            height, width, _ = colour.shape
            mask = np.concatenate((mask, np.ones((height, width, 1))), axis=2)
            greyscale = np.concatenate((greyscale, np.ones((height, width, 1)) * 255), axis=2)

    # Create visual image with foreground color and background grayscale
    visual = np.where(mask, colour, greyscale).astype(np.uint8)

    saveImage(args.visual_path, visual)

if __name__ == "__main__":

    # Create parser for command line arguments
    argParser = argparse.ArgumentParser(prog="segment.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument("-i", "--in", dest="input_path", type=str, required=True, help="Path to input image file")
    argParser.add_argument("-o", "--out", dest="output_path", type=str, default="mask.png", help="Path to which the output mask image file will be saved to")
    argParser.add_argument("-v", "--visual", dest="visual_path", type=str, default="visual.png", help="Path to which the visual image file will be saved to")
    argParser.add_argument("-c", "--cutoff", dest="enhance_cutoff", type=float, default=0.05,
                           help="Cutoff frequency control for high-pass filter. Lower values give stronger high-pass, higher values give gentler filtering. Default = 0.05")
    argParser.add_argument("-s", "--sigma", dest="enhance_sigma", type=float, default=4.0,
                           help="Sigma multiplier for standard deviation in high-pass filter. Larger values give smoother transitions, smaller values give sharper transitions. Default = 4.0")
    argParser.add_argument("-l", "--lower", dest="lower_threshold", type=int, default=80, help="Lower percentile threshold for weak edges, default = 80")
    argParser.add_argument("-u", "--upper", dest="upper_threshold", type=int, default=95, help="Upper percentile threshold for strong edges, default = 95")
    argParser.add_argument("-k", "--kernel", dest="closing_kernel_size", type=int, default=3, 
                           help="Kernel size (must be odd) for closing gaps with dilation and erosion. Larger values close bigger gaps but will thicken edges and risk losing distinct boundaries. Default = 3")
    
    # Parse command line arguments
    args = argParser.parse_args()

     # Validate
    if args.enhance_cutoff < 0:
        sys.exit("Cutoff must be not be negative.")
    elif args.enhance_sigma < 0:
        sys.exit("Sigma must be not be negative.")
    elif args.lower_threshold < 0 or args.lower_threshold > 100:
        sys.exit("Lower threshold must be between 0 and 100.")
    elif args.upper_threshold < 0 or args.upper_threshold > 100:
        sys.exit("Upper threshold must be between 0 and 100.")
    elif args.closing_kernel_size % 2 == 0:
        sys.exit("Closing kernel size must be odd.")
    elif args.closing_kernel_size < 3:
        sys.exit("Closing kernel size must be at least 3.")

    main(args)