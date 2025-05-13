# A Quick Primer on Machine Learning in Wireless Communications

![](https://github.com/farismismar/deepwireless/blob/main/run_video.gif)

DeepWireless is a Python library that allows its user to generate statistics of a MIMO-OFDM wireless channel and both QAM and QPSK symbols (Gray coded).  This library implements what is discussed in my tutorial "A Quick Primer on Machine Learning in Wireless Communications".  Because this tutorial is intended for a single OFDM symbol and a single user, the code can run on CPUs instead of expensive GPU environments.

MIMO precoding is done via singular value decomposition (SVD) or SVD plus waterfilling for spatial multiplexing, identity precoding, and beamforming.  Grid of beams beamforming (i.e., through the use of a discrete Fourier transform (DFT) codebook as a precoder) is also implemented.

Several fading channels are implemented: AWGN, Ricean (and Rayleigh), and 3GPP CDL-A, CDL-C, and CDL-E channels.  The channels are normalized such that their power gain is equal to one.  The large scale gain and shadow fading are implemented per OFDM subcarrier.  There are several models for the large scale gain: (1) free-space pathloss and (2) 3GPP UMa and RMi models.  Support for DeepMIMO has also been added in this most recent version.

Instead of random bit generation for payload, an option to upload 8-bit 32x32 bitmap pictures is now introduced with a plot showing the impact of the channel onto the bitmap.

Several channel operations are performed: pilot-aided channel estimation, channel equalization, and symbol detection.

Channel estimation is performed through the use of least squares estimation and the linear minimum mean square error (LMMSE) formula.  Perfect channel estimation also exists.

Channel state information compression using autoencoders is also implemented.

Channel equalization is performed through the choices of zero forcing and MMSE equalizers.

Symbol detection is performed through four different algorithms: (1) maximum likelihood (2) ensemble learners (random forest) (3) unsupervised learning, and (4) deep feedforward neural networks.

Symbol quantization through truncation is also implemented as an optional step.

Several deep learners are also implemented to facilitate cases as outlined in the reference below.  These deep learner implementations are (1) deep forward neural network (2) long short-term memory deep neural network, and (3) convolutional neural network.

Statistics reported are the channel estimation mean squared error, transmit signal to noise ratio, receive signal to interference plus noise ratio pre- and post-equalization, transmit Eb/N0, receive Eb/N0, bit error rate, pathloss, received signal power, compression loss, and block error rate (BLER).  For BLER, a cyclic redundancy check (CRC) generator polynomial is necessary.  No forward error correction is implemented yet in this simulator.

## Citation

If you use this library (or any code segments thereof) in a scientific publication, we would appreciate citations to the this preprint to keep track of the impact.

`@misc{mismar2023quick, title={A Quick Primer on Machine Learning in Wireless Communications}, author={Faris B. Mismar}, year={2023}, month=dec, eprint={2312.17713}, archivePrefix={arXiv}, primaryClass={cs.NI}}`

## Versioning

| Version        | Date           | Description  |
| ------------- |:-----------------:| :-----------------------|
| 0.1      | 2022-01-02 | First release. |
| 0.2      | 2023-12-28 | Implementation of Primer release of 2023-12-28. |
| 0.3      | 2024-01-03 | Major revision and error fixes. |
| 0.4      | 2024-01-13 | More features. |
| 0.5      | 2024-07-29 | More revisions and 3gpp CDL-C and CDL-E channels. |
| 0.6      | 2024-10-11 | Simplified implementation and more features. |
| 0.61     | 2024-10-15 | Implementation of Primer release of 2024-10-15. |
| 0.7      | 2024-12-07 | Final release for Primer 2024-12-07. |
| 0.8      | 2025-03-19 | Minor feature improvements. |
| 0.9.3    | 2025-05-15 | Class-based implementation, support for more channels and more pathloss models. |
