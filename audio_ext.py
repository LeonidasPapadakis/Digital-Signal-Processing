"""
Spectral Envelope Transfer (Cross-Synthesis)
Extension of equaliser.py
Author: Leonidas Papadakis

This function implements cross-synthesis by transferring the spectral envelope 
(formants / timbre) of a modulator signal onto a carrier signal while preserving
the carrier's fine spectral structure (pitch / excitation).    

Algorithm:
1. Split audio into overlapping frames
2. For every frame:
   3. Compute FFT of both carrier and modulator
   4. Extract log magnitude of modulator
   5. Compute real cepstrum with IFFT
   6. Low-pass filter in quefrency domain (cepstral liftering) to isolate the spectral envelope
   7. Return envelope to frequency domain
   8. Replace the carrier's magnitude with modulator's envelope, keeping carrier's phase.
   9. Inverse FFT and overlap add

Cepstral liftering was chosen because it accurately disentangles the envelope from excitation, 
whereas a frequency domain approach such as moving average smoothing cannot and gives comb artefacts.

Can be sensitive to both the lifter argument and the harmonics of the carrier. If the carrier is percussive 
or contains noise, the excitation division can become unstable and produce artefacts.

A mix parameter is given for blending: 0.0 = original carrier, 1.0 = full transfer

The carrier should be harmonically rich.
The modulator should have a clear, slowly-varying spectral envelope (voices tend to work best).
For example:
    Modulator: human voice (talking, singing, whispering)
    Carrier: piano, guitar, violin music 

Command-line usage:
    python audio_ext.py -c carrier.wav -m modulator.wav -o output.wav
    python audio_ext.py -c violin.wav -m vocals.wav -o crossed.wav -p plot.png --lifter 150 --mix 0.7

Help: python audio_ext.py -h

Dependencies: 
    numpy: array processing and FFT
    scipy: audio resampling
    matplotlib: only imported when plotting
"""
import sys
import argparse
import numpy as np
from equaliser import loadAudio, saveAudio, overlapframes, processFrames, normalise
from scipy.signal import resample

def plotSpectra(fileName: str, carrier: np.ndarray, modulator: np.ndarray,
                output: np.ndarray, sampleRate: int) -> None:
    """
    Plot and save the long term average magnitude spectra of the carrier,
    modulator, and output signals for visual inspection of the envelope transfer.

    The three spectra are plotted on a log-frequency axis in dB and normalises
    so that the peak of each signal is at 0. This makes it easier to compare
    the three signals' envelopes.
    
    The plot is saved to the specified file name.

    Arguments:
        fileName : str
            path to save the plot image
        carrier : np.ndarray
            sample array of the carrier
        modulator : np.ndarray
            sample array of the modulator
        output : np.ndarray
            sample array of the processed output
        sampleRate : int
            common sample rate of the three signals in Hz
    """
    import matplotlib.pyplot as plt

    signals = [carrier, modulator, output]
    labels = ["Carrier", "Modulator", "Output"]

    # Smooth and plot spectra
    smoothingBins = 50
    for signalIndex in range(len(signals)):

        # Get signal and Label
        signal = signals[signalIndex]
        label = labels[signalIndex]
        samples = len(signal)

        # Compute and plot Spectrum
        frequencies = np.fft.rfftfreq(samples, 1 / sampleRate)
        magnitude = np.abs(np.fft.rfft(signal))
        magnitudeDb = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-8) + 1e-8)

        # Smooth Spectrum for better visualisation
        smoothed = np.convolve(magnitudeDb, np.ones(smoothingBins) / smoothingBins, mode="same")
       
        plt.semilogx(frequencies, smoothed, label=label, linewidth=0.8)

    plt.xlim(10, 22000)
    plt.ylim(np.min(smoothed) - 5, np.max(smoothed) + 5)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, normalised)")
    plt.title("Result of spectral envelope transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fileName, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Spectral comparison plot saved to {fileName}")

def extractEnvelope(magnitude: np.ndarray, lifter: int) -> np.ndarray:
    """
    This function extracts the smooth spectral envelope from a magnitude 
    spectrum using cepstral liftering. A spectrum of a signal can be modelled 
    as the product of a slow varying spectral envelope E(f) (timbre) and a 
    rapidly varying fine structure H(f) (pitch)
    |X(f)| = E(f) * H(f)

    Taking the logarithm converts this multiplication into addition:
    log|X(f)| = log E(f) + log H(f). 

    Applying an inverse FFT then maps this into the cepstrum (quefrency domain), 
    where the envelope occupies low quefrency coefficients and the fine structure 
    occupies high quefrency coefficients.

    Uses a low-pass lifter isolates the envelope cleanly using
    homomorphic filtering.
    
    Sensitive to the lifter argument, too low gives an overly smooth envelope,
    too high introduces artefacts.

    Parameters:
        magnitude : np.ndarray
            Array of positive magnitude values from rfft, length frameSize // 2 + 1.
        lifter : int
            Quefrency sample cutoff. Higher values give more harmonic details 
            in the envelope, lower values give smoother.
    Returns:
        np.ndarray
            Smooth linear spectral envelope, same shape as magnitude.
    """

    # Take the log of the magnitude spectrum
    logMagnitude = np.log(magnitude + 1e-10)

    # Compute the real cepstrum with inverse real FFT
    cepstrum = np.fft.irfft(logMagnitude)

    # Apply low-pass lifter
    envelopeCepstrum = np.zeros_like(cepstrum)
    envelopeCepstrum[:lifter]  = cepstrum[:lifter]
    envelopeCepstrum[-lifter:] = cepstrum[-lifter:]

    # Transform back to frequency domain and exponentiate
    envelope = np.exp(np.fft.rfft(envelopeCepstrum).real)

    return envelope

def crossSynthesis(frameList: list[np.ndarray], lifter: int, mix: float) -> np.ndarray:
    """
    This function performs cross-synthesis on the carrier and modulator signals.
    
    The spectral envelope of the modulator is extracted with cepstral liftering, 
    before being imposed onto the carrier's spectral envelope.
    
    The carrier's phase is kept constant so that the output retains the 
    original pitch and rhythm.
    
    The pure excitation signal is isolated by dividing out the carrier's 
    spectral envelope.

    The window applied by overlapframes is undone at the start and
    re-applied at the end so that cepstral extraction operates on the actual
    unwindowed spectrum.
    
    Parameters:
        frames : list[np.ndarray]
            List of two windowed time-domain frames:
                frames[0] — carrier frame (pitch)
                frames[1] — modulator frame (timbre)
        sampleRate : int
            Common sample rate of both signals in Hz.
        lifter : int
            Quefrency sample cutoff. Higher values give more harmonic details 
            in the envelope, lower values give smoother.
        mix : float
            input:output ratio for blend ratio. 
            0.0 = original carrier unchanged,
            1.0 = full envelope transfer. 

    Returns:
        np.ndarray
            Time-domain output signal of length frameSize
    """

    # Unpack frames and get frame size
    carrierFrame = frameList[0]
    modulatorFrame = frameList[1]
    frameSize = len(carrierFrame)

    # Undo the analysis window applied by overlapframes so that the FFT uses the raw signal.
    window = np.hanning(frameSize)
    safeWindow = np.maximum(window, 1e-8)
    carrierFrame   = carrierFrame / safeWindow
    modulatorFrame = modulatorFrame / safeWindow

    # Transform both frames to the frequency domain using real FFT
    cSpectrum = np.fft.rfft(carrierFrame)
    mSpectrum = np.fft.rfft(modulatorFrame)

    # Separate magnitude and phase. A very small value is added to the magnitudes
    # to prevent log10 of 0 (undefined) in extractEnvelope.
    cMagnitude = np.abs(cSpectrum)
    mMagnitude = np.abs(mSpectrum)
    cPhase = np.angle(cSpectrum)

    # Extract the spectral envelope of signals with cepstral liftering
    lifter = max(lifter, 2)
    mEnvelope = extractEnvelope(mMagnitude, lifter)
    cEnvelope = extractEnvelope(cMagnitude, lifter)

    # Divide the carrier's envelope out of its magnitude spectrum to
    # isolate the excitation signal
    cExcitation = cMagnitude / (cEnvelope + 1e-10)

    # Multiply carrier's excitation with the modulator's envelope to impose
    # the modulator's timbre onto the carrier's pitch
    newMagnitude = mEnvelope * cExcitation

    # Recombine with the original carrier phase to preserve pitch and rhythm
    outputSpectrum = newMagnitude * np.exp(1j * cPhase)

    # Blend the output with the original carrier if mixing
    if mix < 1.0:
        outputSpectrum = mix * outputSpectrum + (1.0 - mix) * cSpectrum

    # Re-apply the window before returning
    return np.fft.irfft(outputSpectrum) * window


def main(args: argparse.Namespace) -> None:

    # Load audio files
    carrier, carrierSampleRate = loadAudio(args.carrier_path)
    modulator, modulatorSampleRate = loadAudio(args.modulator_path)

    # Resample if sample rates differ
    if carrierSampleRate != modulatorSampleRate:
        if carrierSampleRate > modulatorSampleRate:
            sampleRate = carrierSampleRate
            modulator = resample(modulator, int(len(modulator) * sampleRate / modulatorSampleRate))
        else:
            sampleRate = modulatorSampleRate
            carrier = resample(carrier, int(len(carrier) * sampleRate / carrierSampleRate))
    else:
        sampleRate = carrierSampleRate

    # Wrap shorter signal to same length
    length = max(len(carrier), len(modulator))
    carrier = np.pad(carrier, (0, length - len(carrier)), mode="wrap")
    modulator = np.pad(modulator, (0, length - len(modulator)), mode="wrap")

    # Compute frame hop
    frameHop = args.frame_size // args.overlap_factor

    # Split both signals into frames
    print("Splitting carrier and modulator into frames...")
    carrierFrames, windowSum = overlapframes(carrier, args.frame_size, frameHop)
    modulatorFrames, _ = overlapframes(modulator, args.frame_size, frameHop)

    # Process both frame lists
    print("Applying cross-synthesis to frames...")
    outputSignal = processFrames(length, [carrierFrames, modulatorFrames], args.frame_size, frameHop,
                                 crossSynthesis, (args.lifter, args.mix_ratio))

    # Normalise output
    outputSignal = normalise(outputSignal, windowSum)
    print("Cross-synthesis complete")

    # Save output
    saveAudio(args.output_path, outputSignal, sampleRate)

    # Plot spectra
    if args.plot_path is not None:
        plotSpectra(args.plot_path, carrier, modulator, outputSignal, sampleRate)
    

if __name__ == "__main__":

    # Create parser for command line arguments
    argParser = argparse.ArgumentParser(prog="ausio_ext.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument("-c", "--carrier", dest="carrier_path", type=str, help="Path to input carrier audio file", required=True)
    argParser.add_argument("-m", "--modulator", dest="modulator_path", type=str, help="Path to input modulator audio file", required=True)
    argParser.add_argument("-o", "--out", dest="output_path", type=str, default="output.wav", help="Path to save output audio file to")
    argParser.add_argument("-p", "--plot", dest="plot_path", type=str, help="Path to save plot of carrier, modulator and output spectra")
    argParser.add_argument("-l", "--lifter", dest="lifter", type=int, default=500, help="Quefrency sample cutoff. Higher values give more harmonic details \
                            in the envelope, lower values give smoother envelope, default = 500")
    argParser.add_argument("-mi", "--mix", dest="mix_ratio", type=float, default=1.0, help="Mix ratio: [0, 1], (0 = original carrier, 1 = full transfer)")
    argParser.add_argument("-f", "--frame", dest="frame_size", type=int, default=8192, help="FFT frame size")
    argParser.add_argument("-ov", "--overlap", dest="overlap_factor", type=int, default=4, help="Overlap factor for overlap-add, default = 4 (75%% overlap)")
    
    # Parse command line arguments
    args = argParser.parse_args()

    # Check mix ratio
    if not (0 <= args.mix_ratio <= 1.0):
        sys.exit("Error: --mix must be between 0.0 and 1.0")


    main(args)