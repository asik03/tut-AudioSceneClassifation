# Audio Scene Classifier

The main goal of this project is present a performance comparison when using different features. We will use the same model 
based on a convolutional neural network and we will train it using different features as input. The impact on the 
performance depending on the features used will be surveyed.

## Feature extraction

We will train different networks using the following features, and we will present a comparison.

1. Mel spectogram: Mel-scaled power spectogram
2. MFCC: Mel-Frequency Cepstral Coefficients
3. Chroma STFT: Chromagram from a waveform or power spectrogram
4. Chroma CQT: Constant-Q chromagram
5. Spectral Contrast: Spectral contrast

All the proposed are spectral features based, to extract them we will use the python library [librosa](https://librosa.github.io/librosa/feature.html#spectral-features "librosa").

[Todo] Detail each transformation.

## Convolutional Neural Network

Our CNN model is based in [this reference](http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Han_207.pdf "CNN"), 
and it is as follows

![alt text](./docs/images/CNN_model.png?raw=true "CNN model")

For each stereo audio signal, we will pre-process independently left and right channels extracting the features.
Both features will be the input for two independent CNN, as described in the previous figure, and after that we will
concatenate to estimate the category through a softmax regression.