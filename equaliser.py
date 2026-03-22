"""
11-band frequency-domain equaliser
Author: Leonidas Papadakis

This function implements a multi-band equaliser using frequency-domain
processing.

The equaliser has 11 fixed octave bands covering the audible frequency range.
Fixed band centre frequencies (Hz):
    [6, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

It uses:
- Hann windowing to reduce spectral leakage
- an overlap-add technique to minimize artefacts
- linear interpolation between band gains to for a smooth transition

This function uses FFT-based frequency domain filtering rather than
IIR / FIR filter banks for several reasons:

Advantages:
- Perfect control over magnitude response with no ripple between bands
- No phase distortion - phase response is linear
- Computationally efficient for large files

Disadvantages compared to IIR filters:
- Not suitable for real-time streaming 
- More memory intensive due to frame-based processing
- Higher runtimes for small files

The overlap-add technique with Hann windowing mitigates the main disadvantage
of frequency-domain processing by eliminating discontinuities at frame boundaries,
making this approach suitable for high-quality offline audio processing.

Example usage:
    python equaliser.py -i input.wav -o output.wav -b "0,-1,0,0,-3,0,0,3,0,0,2"
    python equaliser.py -i input.wav -o output.wav -b "0,0,0,0,0,0,0,0,0,0,0" -p response.png
    python equaliser.py -i input.wav -o output.wav -b "0,-3,0,3,0,-3,0,3,0,-3,0" -p plot.png -f 4096 -ov 8

Help: python equaliser.py -h

Dependencies: 
    numpy: array processing and FFT
    soundfile: audio file I/O
    matplotlib: only imported when plotting
"""

import sys
import argparse
import soundfile as sf
import numpy as np

def parseGains(gainsString : str) -> np.ndarray:
    """
    Parses the --bands argument from command line into an array of 11 gains.
    Validates to ensure exactly 11 values are provided.
    Whitespace around each number is automatically stripped.
    
    Args:
        gainsString : str
            Comma-separated string of 11 numbers representing the gain in decibels.
            for each band, e.g. "0,3,3,0,-2,0,0,2,0,1,0".
    Returns:
        np.ndarray
            Array of 11 float values containing the parsed gains in dB.
    """

    # Split gains by comma
    gains = np.array([float(gain.strip()) for gain in gainsString.split(',')])
    
    # Check exactly 11 gains were given
    if len(gains) != 11:
        sys.exit("Error: --bands must contain exactly 11 comma-separated numbers")
    
    return gains

def loadAudio(fileName : str) -> tuple[np.ndarray, int]:
    """
    Reads an audio file and convert to mono.
    
    Loads an audio file using soundfile library. If the audio has multiple
    channels (stereo or more), it is converted to mono by averaging across
    channels.
    
    Args:
        fileName : str
            Path to the input audio file. 
            Supports WAV, MP3, OGG, and other formats supported by soundfile.
    Returns:
        tuple[np.ndarray, int]
            - signal: Numpy array of audio samples.
            - sampleRate: ISample rate of the audio in Hz.
    """

    # Try to read file
    try:
        signal, sampleRate = sf.read(fileName)

        # Force mono
        if signal.ndim == 2:
            signal = np.mean(signal, axis=1)

    except Exception as e:
        sys.exit(e)
    
    return signal, sampleRate

def saveAudio(fileName : str, signal : np.ndarray, sampleRate : int) -> None:
    """
    Writes the output signal to an audio file using soundfile.
    
    Args:
        fileName : str
            Path where the output audio file will be saved.
        signal : np.ndarray 
            1D numpy array of audio samples to save.
            Values should be in range [-1, 1].
        sampleRate : int
            Sample rate of the audio signal in Hz.
    """

    # Try to save file
    try:
        sf.write(fileName, signal, sampleRate)
        print("Equalised audio saved to", fileName)

    except Exception as e:
        sys.exit(e)

def plotFrequencyResponse(fileName, sampleRate, bands, gains) -> None:
    """
    This method generates and save a plot of the equaliser's frequency response.
    
    Creates a semilogarithmic plot showing gains against frequency 
    for the combined equaliser response. The plot helps visualize how the
    equaliser affects different frequency bands and the smoothness of
    transitions between bands.
    Shows frequencies from 10 Hz to 22 kHz (audible range).
    
    Parameters:
        fileName : str
            Path where the plot image will be saved.
        sampleRate : int
            Sample rate used to determine the frequency range.
        bands : np.ndarray
            Array of 11 centre frequencies for each band.
        gains : np.ndarray
            Array of 11 gain values in dB for each band.
    """
    import matplotlib.pyplot as plt

    plots = 10000

    # Calculate frequency response
    gainMask = createMask(sampleRate, plots, bands, gains)
    frequencies = np.fft.rfftfreq(plots, 1 / sampleRate)

    # Create plot
    plt.figure(figsize = (10, 5))
    plt.semilogx(frequencies, 20 * np.log10(gainMask))
    plt.xlim(10, 22000)
    plt.ylim(-25, 25)
    plt.grid(True, which = "both", linestyle = "--", alpha = 1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("11 band Equalizer Frequency Response")

    # Save plot
    plt.savefig(fileName, bbox_inches = "tight")
    plt.close()
    print("Frequency response saved to", fileName)

def createMask(sampleRate : int, frameSize : int, bands : np.ndarray, decibalGains : np.ndarray) -> np.ndarray:
    """
    This function creates a gain mask for the equaliser in the frequency 
    domain.

    Constructs a smooth gain mask by interpolating between the specified
    band gains in log-frequency space. 
    This ensures smooth transitions between bands and avoids abrupt changes 
    that could cause artifacts.
    
    The interpolation is performed in log-frequency space because human
    hearing perceives frequency logarithmically, and equaliser bands are
    spaced logarithmically in octaves.
    
    Parameters:
        sampleRate : int
            Sample rate of the audio signal in Hz.
        frameSize : int
            Number of samples in FFT frame.
        bands : np.ndarray
            Array of 11 centre frequencies for each band in Hz.
        decibelGains : np.ndarray
            Array of 11 gain values in dB for each band.
    Returns:
        np.ndarray
            Gain mask for real FFT.
            Values are linear gains, ready for multiplication with spectrum.
    """
    
    # Frequency axis for real FFT (0 : Nyquist)
    freqencies = np.fft.rfftfreq(frameSize, d = 1 / sampleRate)

    # Make the minimum frequency 1 to avoid log10 of 0 (undefined)
    freqencies = np.maximum(freqencies, 1)

    # Convert frequencies and bands to log scale
    freqencies = np.log10(freqencies)
    bands = np.log10(bands)

    # Convert gains from decibels to linear
    linearGains = 10 ** (decibalGains / 20)

    # Linear interpolation to create mask
    mask = np.interp(freqencies, bands, linearGains, left = linearGains[0], right = linearGains[-1])

    return mask

def applyMask(inputSignals : list[np.ndarray], mask : np.ndarray) -> np.ndarray:
    """
    This function applies the equaliser gain mask to a single frame.
    
    It processes one frame of audio by:
    - Converting to frequency domain using FFT
    - Multiplying the spectrum by the gain mask
    - Converting back to time domain using inverse FFT
    
    Parameters:
        inputSignals : list[np.ndarray]
            List containing a single audio frame. The list format allows 
            compatibility with the callable processer() argument in processFrames()
            which handles multiple signals.
        mask : np.ndarray
            Linear gain mask from createMask() to apply to the spectrum.
    Returns:
        np.ndarray
            Processed time-domain audio frame of the same length as the input frame.
    """
    # Only one set of frames for equaliser
    inputSignal = inputSignals[0]

    # Convert time domain to frequency domain using real FFT
    spectrum = np.fft.rfft(inputSignal)

    # Apply mask
    spectrum *= mask

    # Convert back to time domain using inverse real FFT
    return np.fft.irfft(spectrum)

# Split audio into frames
def overlapframes(inputSignal : np.ndarray, frameSize : int, hop : int) -> np.ndarray:
    """
    Split an audio signal into overlapping frames using Hanning windowing.
    
    Creates a series of overlapping frames from the input signal, each
    multiplied by a Hanning window to reduce spectral leakage. The overlap
    between frames ensures smooth reconstruction after processing.
    
    Parameters:
        inputSignal : np.ndarray
            Complete audio signal.
        frameSize : int
            Number of samples in each frame.
        hop : int
            Number of samples to advance between frame starts for overlapping.
    
    Returns:
        tuple[list[np.ndarray], np.ndarray] containing:
            frames : list[np.ndarray]
                List of windowed audio frames
            windowSum : np.ndarray
                Array of cumulative window contribution at each 
                sample position, used for normalisation after processing
    """

    # Create window to reduce spectral leakage
    window = np.hanning(frameSize)

    # Create empty array to sum applied windows (for normalisation)
    windowSum = np.zeros(len(inputSignal))

    # Create empty array for equalised output
    frames = []

    # Split sample array into frames
    for start in range(0, len(inputSignal), hop):

        # Calculate end of frame
        end = min(start + frameSize, len(inputSignal))
        
        # Extract frame from input signal
        frame = np.zeros(frameSize)
        frame[:end - start] = inputSignal[start:end]

        # Apply window to frame
        frame *= window

        # Append frame to frames array
        frames.append(frame)

        # Add window to window sum
        windowSum[start:end] += window[:end - start]

    return frames, windowSum

def processFrames(inputLength : int, framesList : list[np.ndarray], frameSize : int, frameHop : int, processer : callable, processerArgs : tuple) -> np.ndarray:
    """
    This function processes a list of frame sets with a frame processing function.
    Used with one set of frames in this equaliser to apply the equaliser mask, but made 
    generic for use in cross-synthesis in audio_ext.py.
    
    Parameters:
        inputLength : int
            Original length of the input signal in samples.
        framesList : list[list[np.ndarray]]
            List containing one or more lists of frames to process simultaneously.
        frameSize : int
            Size of each frame in samples.
        frameHop : int
            Hop size between frames in samples.
        processer : callable
            Function that processes one set of frames. Should take (current_frames, *processArgs) 
            and return a frame.
        processerArgs : tuple
            Additional arguments to pass to processer.
    
    Returns:
        np.ndarray
            Reconstructed signal after processing all frames and overlap-adding.
    """
    
    # Create empty array for equalised output
    outputSignal = np.zeros(inputLength)

    # Split sample array into frames
    for frameNumber in range(len(framesList[0])):

        # Calculate start and end of frame
        start = frameNumber * frameHop
        end = min(start + frameSize, inputLength)

        # Get current frame for each frameSet in list
        currentFrames = []
        for i in range(len(framesList)):
            currentFrames.append(framesList[i][frameNumber])

        
        # Process frame
        outputFrame = processer(currentFrames, *processerArgs)
        
        # Add frame to output output signal
        outputSignal[start:end] += outputFrame[:end - start]
    
    return outputSignal

def normalise(signal : np.ndarray, windowSum : np.ndarray) -> np.ndarray:
    """
    Normalise the output signal after overlap-add processing.
    Compensates for the amplitude modulation caused by windowed frames.
    Divides the signal by the cumulative window sum to restore the original amplitude.
    Very small windowSum values are ignored to prevent division by zero or 
    extreme amplification of noise
    
    Parameters:
        signal : np.ndarray
            Signal requiring normalisation.
        windowSum : np.ndarray
            Array of cumulative window contributions from overlapFrames().
    Returns:
        np.ndarray
            Normalised signal.
    """
    # Remove very small values from window sum to avoid division by zero
    windowMask = windowSum > 1e-8

    # Divide output signal by window sum to normalise
    signal[windowMask] /= windowSum[windowMask]

    return signal

def main(args : argparse.Namespace):
    
    # Parse gains for each band from command line argument
    gains = parseGains(args.bands_string)

    # Read input audio
    inputSignal, sampleRate = loadAudio(args.input_path)

    # Define band frequencies
    bands = np.array([6, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    
    # Create mask for equaliser
    print("Creating equaliser mask...")
    mask = createMask(sampleRate, args.frame_size, bands, gains)
    
    # Calculate hop size for overlapping frames
    frameHop = args.frame_size // args.overlap_factor

    # Split audio into frequency domain frames
    print("Splitting into frames...")
    frames, windowSum = overlapframes(inputSignal, args.frame_size, frameHop)

    # Equalise
    print("Applying equaliser to frames...")
    outputSignal = processFrames(len(inputSignal), [frames], args.frame_size, frameHop, applyMask, (mask,))
    
    # Normalise
    outputSignal = normalise(outputSignal, windowSum)
    print("Equalisation complete")

    # Save equalised audio
    saveAudio(args.output_path, outputSignal, sampleRate)
    
    # Plot frequency response
    if args.plot_path is not None:
        plotFrequencyResponse(args.plot_path, sampleRate, bands, gains)
        
    

if __name__ == "__main__":

    # Create parser for command line arguments
    argParser = argparse.ArgumentParser(prog="equaliser.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument("-i", "--in", dest="input_path", type=str, help="Input audio file path", required=True)
    argParser.add_argument("-o", "--out", dest="output_path", type=str, default="output.wav", help="Path to save output audio file")
    argParser.add_argument("-p", "--plot", dest="plot_path", type=str, help="Path to save plot of frequency response")
    argParser.add_argument("-b", "--bands", dest="bands_string", type=str, help="Gain values for each band", required=True)
    argParser.add_argument("-f", "--frame", dest="frame_size", type=int, default=8192, help="FFT frame size")
    argParser.add_argument("-ov", "--overlap", dest="overlap_factor", type=int, default=4, help="Overlap factor, default = 4 (75%% overlap)")
    
    # Parse command line arguments
    args = argParser.parse_args()

    # Validate command line arguments
    if args.frame_size <= 0:
        sys.exit("Frame size must be greater than 0")
    elif args.overlap_factor <= 0:
        sys.exit("Overlap factor must be greater than 0")

    main(args)