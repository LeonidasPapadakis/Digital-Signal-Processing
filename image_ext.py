"""
Gradient-Weighted Spatial K-Means Segmentation
Extension of segment.py
Author: Leonidas Papadakis

This function performs multi-class image segmentation using a custom K-means
algorithm, which incorporates both spatial proximity and gradient magnitude
weighting. It extends binary segmentation by partitioning images into multiple
regions (e.g. grass, sky, objects, shadows).

Standard K-means clustering on color alone fails in several common scenarios:
- Objects with color gradients get fragmented into different regions
- Boundaries become fuzzy and badly aligned with actual edges
- Shapes become flat, losing their textures

This implementation addresses these limitations through:

Spatial Weighting:
    Adding normalized coordinates creates a 5D feature space (R,G,B,x,y).
    This encourages spatial coherence, as pixels near each other are more
    likely to be in the same cluster, helping to prevent fragmentation from 
    color gradients. The spatial_weight parameter is used to scale this effect.

Gradient-Weighted Centroids:
    Pixels with strong gradients (edges) contribute more to cluster centroids.
    This is based on the insight that edge pixels carry crucial information
    about object boundaries. By weighting them more heavily:
    - Cluster boundaries align more precisely with actual edges
    - Segmentation quality improves around object boundaries

Limitations:
    - Still assumes circular clusters just like K-Means (from Euclidean distance)
    - Gradient weighting may over-emphasize noisy edges

Usage:
    python image_ext.py -i input.jpg -o clusters.png
    python image_ext.py -i input.jpg -o clusters.png -k 10 -sw 2.5 -c 0.03 -t 100
    
    Help:
        python image_ext.py -h

Dependencies:
    numpy: Array processing
"""
import argparse, sys
import numpy as np
from segment import loadImage, saveImage, enhance, sobelGradients

def addSpatialFeatures(image : np.ndarray, spatialWeight : float = 1.5) -> np.ndarray:
    """
    This function concatenates an array of pixels with normalized (x, y) spatial coordinates.
    This creates a 5D feature vector per pixel: [R, G, B, xNorm, yNorm]
    
    This enables clustering to consider both color similarity and spatial proximity simultaneously.
    
    The spatial coordinates are normalized to [0, 1] range and then scaled by
    spatialWeight. Higher spatialWeight values make spatial proximity more
    important than color similarity, giving more compact regions.
    Spatial coordinates are normalized to [0,1] to make them dimensionally
    comparable to RGB values, which are also normalized to [0, 1]
    
    Parameters:
        image : np.ndarray
            Input image in RGB format.
        spatialWeight : float, default=1.5
            Scaling factor for spatial dimensions.
    Returns:
        np.ndarray
            Feature array of shape (H, W, 5).
    """
    height, width, _ = image.shape
    
    # Normalize spatial coordinates to scale [0, 1]
    y, x = np.mgrid[0 : height, 0 : width] 
    xNormalised = x.astype(np.float64) / (width - 1)
    yNormalised = y.astype(np.float64) / (height - 1)
    
    # Scale spatial part
    xScaled = xNormalised * spatialWeight
    yScaled = yNormalised * spatialWeight
    
    # Stack spatial features to fit shape (height, width, 2)
    spatial = np.stack([xScaled, yScaled], axis=-1)
    
    # Combine with color image to shape (height, width, channels + 2)
    features = np.concatenate([image.astype(np.float64), spatial], axis=-1)
    
    return features

def weightedCluster(spatialFeatures : np.ndarray, magnitudes : np.ndarray, k : int, maxIterations : int, trials : int) -> tuple[np.ndarray, np.ndarray]:
    """
    This function performs spatial K-means clustering with gradient-weighted centroid computation.
    This is the core segmentation algorithm that extends standard K-means by using gradient
    magnitude to add weight to pixels. Pixels with stronger gradients (edges) contribute more 
    to cluster centroids, causing boundaries to align better with actual image edges.
    
    The algorithm is as follows:
    1 - Compute gradient weights: (magnitude / max_magnitude) ^ 2
    2 - Initialize centroids randomly
    3 - Run spatial K-means 
    4 - Compute new centroids weighted by gradient magnitude
    4 - Repeat for multiple random initializations and choose best result
    
    The squared weighting of gradient magnitude emphasizes strong edges even further,
    creating a non-linear emphasis on boundary pixels.
    
    Parameters:
        spatialFeatures : np.ndarray
            Input spacial featured array from addSpatialFeatures().
        magnitudes : np.ndarray
            Gradient magnitudes from sobelGradients().
            Higher values indicate stronger edges.
        spatialWeight : float
            Weight for spatial features in clustering.
            Higher values = more compact, spatially coherent regions.
        k : int
            Number of clusters to create.
        maxIterations : int, default=100
            Maximum iterations per trial.
            May stop earlier if convergence is detected.
        trials : int, default=5
            Number of random initializations. The result with lowest inertia is kept.
    Returns:
        tuple[np.ndarray, np.ndarray] containing:
            cluster_map : np.ndarray
                2D array with integer cluster assignments (0 to k-1).
                Each pixel value indicates which region it belongs to.
            centroids : np.ndarray
                Final cluster centroids containing only the RGB components.
    """
    # Check cluster count is valid
    if k < 2:
        sys.exit("Error: Must have at least 2 clusters")

    # Get dimensions
    rows, cols, channels = spatialFeatures.shape
    pixels = rows * cols

    # Flatten to height * width, channels and cast to float
    image = spatialFeatures.reshape(-1, channels).astype(np.float64)

    # Give higher weight to stronger gradients
    maxGradient = magnitudes.max()
    if maxGradient > 0:
        gradientWeights = (magnitudes.reshape(-1) / maxGradient) ** 2
        
    else:
        # If no strong gradients, give all equal weight
        gradientWeights = np.ones(pixels)

    # Track best clusters
    bestInertia = np.inf
    bestClusters = None
    bestCentroids = None

    # Run K-Means for multiple random initializations
    for _ in range(trials):

        # Initialize cluster centroids randomly (better than pure random: sample from data)
        initialIndexes = np.random.choice(pixels, k, replace = False)
        centroids = image[initialIndexes].astype(np.float64)
        
        # Improve clusters until clusters converge within tolerance or max iterations reached
        for _ in range(maxIterations):
            
            # Compute distances from each point to each centroid
            distances = np.sqrt(((image[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2).sum(axis=2))
            
            # Assign each point to the cluster with the closest centroid
            clusters = np.argmin(distances, axis=1)
            
            # Update centroids
            newCentroids = np.zeros((k, channels), dtype=np.float64)
            
            # Compute new centroids for each cluster
            for cluster in range(k):
                
                # Get cluster mask
                mask = (clusters == cluster)
                
                # If cluster is not empty
                if mask.sum() > 0:
                    
                    # Get pixels in this cluster
                    clusterPixels = image[mask]
                    
                    # Get corresponding gradient weights for these pixels
                    clusterWeights = gradientWeights[mask]
                    
                    # Pixels with stronger gradients contribute more to the centroid
                    if clusterWeights.sum() > 0:

                        # Normalize weights to sum to 1
                        normalisedWeights = clusterWeights / clusterWeights.sum()
                        
                        # Compute weighted centroid
                        newCentroids[cluster] = np.sum(clusterPixels * normalisedWeights[:, np.newaxis], axis=0)
                    else:
                        # Update centroid with simple mean if all weights are zero
                        newCentroids[cluster] = clusterPixels.mean(axis=0)
                else:
                    # Re-initialize to random point
                    newCentroids[cluster] = image[np.random.choice(pixels)]
            
            # Compute inertia - sum of squared distances from each point to its assigned centroid
            inertia = ((image - newCentroids[clusters])**2).sum()
            
            # Compute shift between new and old centroids
            shift = np.sqrt(((newCentroids - centroids)**2).sum(axis=1)).max()
            centroids = newCentroids
            
            # Check for convergence
            if shift < 1e-4:
                break
        
        # Check if this trial gave the best inertia
        if inertia < bestInertia:
            bestInertia = inertia
            bestClusters = clusters
            bestCentroids = centroids

    # Reshape back to original shape
    bestClusters = bestClusters.reshape(rows, cols)

    # Remove spatial features from centroids
    bestCentroids = bestCentroids[:, :channels-2]

    return bestClusters, bestCentroids


def main(args : argparse.Namespace):

    # Read input image and convert to grayscale
    colour, greyscale = loadImage(args.input_path)

    # Apply FFT high-pass filter
    print("Enhancing image...")
    enhanced = enhance(greyscale, args.enhance_cutoff, args.enhance_sigma)

    # Compute gradient magnitude for weighted K-Means
    print("Computing Sobel gradients...")
    magnitudes, _ = sobelGradients(enhanced)

    # Add spatial features
    print("Adding spatial features...")
    spatialFeatures = addSpatialFeatures(colour, args.spatial_weight)

    # Run weighted K-Means
    print(f"Clustering into {args.cluster_count} regions...")
    clusters, centroids = weightedCluster(spatialFeatures, magnitudes, args.cluster_count, args.max_iterations, args.trials)

    print("Segmentation complete.")
    
    # Save clustered image
    clusterColours = centroids[clusters].astype(np.uint8)
    saveImage(args.output_path, clusterColours)


if __name__ == "__main__":

    # Create parser for command line arguments
    argParser = argparse.ArgumentParser(prog="image_ext.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument("-i", "--in", dest="input_path", type=str, required=True, help="Path to input image file")
    argParser.add_argument("-o", "--out", dest="output_path", type=str, default="output.png", help="Path to which the clustered output image will be saved to")
    argParser.add_argument("-k", "--clusters", dest="cluster_count", type=int, default=5, help="Number of regions to cluster the image into, default = 5")
    argParser.add_argument("-m", "--max_iter", dest="max_iterations", type=int, default=100, help="Maximum number of iterations for k-means, default = 100")
    argParser.add_argument("-t", "--trials", dest="trials", type=int, default=5, help="Number of trials for k-means, default = 10")
    argParser.add_argument("-c", "--cutoff", dest="enhance_cutoff", type=float, default=0.05,
                           help="Cutoff frequency control for high-pass filter. Lower values give stronger high-pass, higher values give gentler filtering. Default = 0.05")
    argParser.add_argument("-s", "--sigma", dest="enhance_sigma", type=float, default=4.0,
                           help="Sigma multiplier for standard deviation in high-pass filter. Larger values give smoother transitions, smaller values give sharper transitions. Default = 4.0")
    argParser.add_argument("-l", "--lower", dest="lower_threshold", type=int, default=80, help="Lower percentile threshold for weak edges, default = 80")
    argParser.add_argument("-u", "--upper", dest="upper_threshold", type=int, default=95, help="Upper percentile threshold for strong edges, default = 90")
    argParser.add_argument("-sw", "--spatial_weight", dest="spatial_weight", type=float, default=5.0, help="Weighting factor of spatial features used in hybrid clustering, default = 5.0")

    # Parse command line arguments
    args = argParser.parse_args()

    # Validate
    if args.cluster_count < 2:
        sys.exit("Number of clusters must be at least 2.")
    elif args.enhance_cutoff < 0:
        sys.exit("Cutoff must be not be negative.")
    elif args.enhance_sigma < 0:
        sys.exit("Sigma must be not be negative.")
    elif args.lower_threshold < 0 or args.lower_threshold > 100:
        sys.exit("Lower threshold must be between 0 and 100.")
    elif args.upper_threshold < 0 or args.upper_threshold > 100:
        sys.exit("Upper threshold must be between 0 and 100.")
    elif args.spatial_weight < 0:
        sys.exit("Spatial weight must be non-negative.")
    elif args.max_iterations < 0:
        sys.exit("Maximum iterations must be non-negative.")
    elif args.trials < 0:
        sys.exit("Number of trials must be non-negative.")

    main(args)
