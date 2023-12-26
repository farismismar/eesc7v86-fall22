# EESC 7v86: Machine Learning in Wireless Communications

This is a code repository that allows its user to generate statistics of a multiple input, multiple output (MIMO) wireless channel with Rayleigh fading and both QAM and QPSK symbols (Gray coded).

Several channel operations are performed: channel estimation, channel equalization, and symbol detection.

Channel estimation is performed through the use of least squares and linear minimum mean squared error (L-MMSE) estimation formulas.

Channel equalization is performed through the choices of zero forcing and minimum mean squared error (MMSE) equalizers.

Symbol detection is performed through four different algorithms: 1) maximum likelihood 2) unsupervised learning 3) fully connected deep neural networks and 4) ensemble learning.

Statistics reported are the channel estimation mean squared error, signal to noise ratio (both TX and RX), Eb/N0 (both TX and RX), symbol error rate per stream, bit error rate per stream, and block error rate.

## Citation

If you use this codes (or any code segments thereof) in a scientific publication, we would appreciate citations to the following paper:

`@misc{mlwirelesscomm,
title={A Tutorial on Machine Learning in Wireless Networks},
author={Mismar, F. B.},
year={2022},
eprint={220x.yyyy},
archivePrefix={arXiv},
primaryClass={cs.IT}
}`


## Versioning

| Version        | Date           | Description  |
| ------------- |:-------------:| :-----|
| 0.1      | 2022-01-02 | First release |
| 0.2      | 2022-02-24 | Added k-means detection |
| 0.3      | 2022-03-07 | Added unsupervised detection and M-QAM |
| 0.4      | 2023-12-26 | Pilot-based channel estimation |
