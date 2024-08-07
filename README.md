# EESC 7v86 Selected Topics in Wireless Communications: Deep Learning for 5G

This is a code repository that allows its user to generate statistics of a multiple input, multiple output (MIMO) wireless channel with Rayleigh fading and both QAM and QPSK symbols (Gray coded).

Several channel operations are performed: channel estimation, channel equalization, and symbol detection.

Channel estimation is performed through the use of least squares estimation formula.

Channel equalization is performed through the choices of ZF and MMSE equalizers.

Symbol detection is performed through three different algorithms: 1) maximum likelihood 2) unsupervised learning and 3) fully connected deep neural networks.

Statistics reported are the channel estimation mean squared error, signal to noise ratio (both TX and RX), Eb/N0 (both TX and RX), symbol error rate per stream, bit error rate per stream, pathloss, and block error rate.

The additional file `PlottingUtils.py` is a class used to plot probability density functions (PDF) and cumulative distribution functions (CDF) for various configurations (joint, marginal) PDFs and CDFs, box plots, and PDFs/CDFs for mulitple variables on a single plot.

The preprint updates will be uploaded here under the name `primer.pdf`

## Citation

If you use this codes (or any code segments thereof) in a scientific publication, we would appreciate citations to the this preprint to keep track of the impact.

`@misc{mismar2023quick, title={A Quick Primer on Machine Learning in Wireless Communications}, author={Faris B. Mismar}, year={2023}, month=dec, eprint={2312.17713}, archivePrefix={arXiv}, primaryClass={cs.NI}}`

## Versioning

| Version        | Date           | Description  |
| ------------- |:-------------:| :-----|
| 0.1      | 2022-01-02 | First release |
| 0.2      | 2023-12-28 | Implementation of Primer release of 2023-12-28 |
| 0.3      | 2024-01-03 | Major revision and error fixes. |
| 0.4      | 2024-01-13 | More features. |
| 0.5      | 2024-07-29 | More revisions and 3gpp CDL-C and CDL-E channels. |
