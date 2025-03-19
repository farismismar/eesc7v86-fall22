# A Quick Primer on Machine Learning in Wireless Communications

![](https://github.com/farismismar/eesc7v86-fall22/blob/main/run_video.gif)

This is a code repository that allows its user to generate statistics of a MIMO-OFDM wireless channel and both QAM and QPSK symbols (Gray coded).  The simulation is for a single OFDM symbol and a single user.

MIMO precoding is done via singular value decomposition (SVD), waterfilling, and identity.  Beamforming through the use of discrete Fourier transform (DFT) codebook is implemented.

Several fading channels are implemented: AWGN, Ricean (and Rayleigh), and 3GPP CDL-A, CDL-C, and CDL-E channels.  The channels are normalized such that their power gain is equal to one.  The large scale gain and shadow fading are implemented per OFDM subcarrier.

Several channel operations are performed: channel estimation, channel equalization, and symbol detection.

Channel estimation is performed through the use of least squares estimation and the linear minimum mean square error (LMMSE) formula.

Channel equalization is performed through the choices of zero forcing and MMSE equalizers.

Symbol detection is performed through four different algorithms: 1) maximum likelihood, 2) ensemble learners (forest), 3) unsupervised learning, and 4) deep neural networks.

Symbol quantization through Lloyd-Max is also implemented from a forked implementation (credits in source code).  Quantization is optional.

Statistics reported are the channel estimation mean squared error, signal to noise ratio (both TX and RX), TX Eb/N0, bit error rate, pathloss, and block error rate.

The additional file `PlottingUtils.py` is a class used to plot probability density functions (PDF) and cumulative distribution functions (CDF) for various configurations (joint, marginal) PDFs and CDFs, box plots, and PDFs/CDFs for mulitple variables on a single plot.

The preprint updates will be uploaded to [arXiv](https://arxiv.org/pdf/2312.17713).  The code is best run on Python `3.9.13` on Windows with TensorFlow `2.10.0` with GPU support with a supported NVIDIA GPU running cuDNN `9.5.0` and cuda `12.6.2`. 

## Teaching

I use this code in my course EESC 7v86: Special Topics in Wireless Communications: Deep Learning in 5G.  I offer and teach this course at UT-Dallas on part-time basis whenever I have the availability and adequate student interest.

## Citation

If you use this code (or any code segments thereof) in a scientific publication, we would appreciate citations to the this preprint to keep track of the impact.

`@misc{mismar2023quick, title={A Quick Primer on Machine Learning in Wireless Communications}, author={Faris B. Mismar}, year={2023}, month=dec, eprint={2312.17713}, archivePrefix={arXiv}, primaryClass={cs.NI}}`

## Versioning

| Version        | Date           | Description  |
| ------------- |:-------------:| :-----|
| 0.1      | 2022-01-02 | First release. |
| 0.2      | 2023-12-28 | Implementation of Primer release of 2023-12-28. |
| 0.3      | 2024-01-03 | Major revision and error fixes. |
| 0.4      | 2024-01-13 | More features. |
| 0.5      | 2024-07-29 | More revisions and 3gpp CDL-C and CDL-E channels. |
| 0.6      | 2024-10-11 | Simplified implementation and more features. |
| 0.61     | 2024-10-15 | Implementation of Primer release of 2024-10-15. |
| 0.7      | 2024-12-07 | Final release for Primer 2024-12-07. |
| 0.8      | 2025-03-19 | Minor feature improvements. |
