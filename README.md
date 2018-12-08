# Audio Scene Classifier

## Feature extraction

We will train different networks using the following features, and we will present a comparison.

1. Mel spectogram: Mel-scaled power spectogram
2. MFCC: Mel-Frequency Cepstral Coefficients
3. Chorma STFT: Chromagram from a waveform or power spectrogram
4. Spectral Contrast: Spectral contrast
5. Tonnetz: Tonal centroid features (tonnetz)

All the proposed are spectral features based, to extract them we will use the python library [librosa](https://librosa.github.io/librosa/feature.html#spectral-features "librosa").

[Todo] Detail the dimension of each transformation.

## Convolutional Neural Network

Our CNN model is based in [this reference](http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Han_207.pdf "CNN"), 
and it is as follows

![alt text](https://bitbucket.org/tut4win/audio-scene-classifier/src/master/docs/images/CNN_model.png "CNN model")
