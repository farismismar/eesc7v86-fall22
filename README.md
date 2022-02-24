# Machine Learning in Wireless Communications

This is a code repository that allows its user to generate statistics of a multiple input, multiple output (MIMO) wireless channel with Rayleigh fading and QPSK symbols.  Currently, the precoder is set to the identity matrix.

Several channel operations are performed: channel estimation, channel equalization, quantization, and symbol detection.

Channel estimation is performed through the use of least squares and linear regression machine learning technique.

Channel equalization is performed through the choices of zero forcing and minimum mean squared error (MMSE) equalizers.

The current quantization supported at the moment is $b = 1$ (besides no quantization or $b = \inf$).

Symbol detection is performed through four different algorithms: 1) maximum likelihood 2) unsupervised learning 3) fully connected deep neural networks and 4) ensemble learning.

Statistics reported are the channel estimation mean squared error, symbol error rate, bit error rate, and block error rate.

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
| ------------- |:-------------:| -----:|
| 1.0      | 2022-01-02 | First release |
| 1.1      | 2022-02-24 | Added k-means detection |
