# Macroeconomic Uncertainty Prices when Beliefs are Tenuous

This repository contains code which estimates the empirical model in "Macroeconomic Uncertainty Prices when Beliefs are Tenuous" by [Lars Peter Hansen][id1] and [Thomas J Sargent][id2]. Appendix B and C from that paper outline the methodology employed in this software.

[id1]: https://larspeterhansen.org/
[id2]: http://www.tomsargent.com/

## Getting Started

To copy the code to your machine, you may either download it from the Github website directly or you may clone the repository in read-only mode.

### Prerequisites

This project simply requires the Anaconda distribution of Python version 3.x. Additional dependencies and prerequisites are handled automatically in setup.

### Installing

Navigate to the folder containing the code and run the `setup.sh` script by entering

For Mac Users, please run 
```
source setup.sh
```
For Windows Users, please run
```
conda update conda
conda env create -f environment.yml
conda activate tenuous
```

Press `y` to proceed with installation when prompted. You will know that setup has been correctly implemented if the word `(tenuous)` contained in parenthesis appears on the current line of your terminal window.

## VAR Estimation

### Running the estimation

To run the code, simply use

```
python tenuous_estimation.py
```

This code will print in terminal the estimated 10th, 50th, and 90th percentiles for the data. The results printed as weighted percentiles should be close to the results listed in Appendix B, with variations in random number generation accounting for any differences. The code will also produce a histogram for each of the relevant parameters, showing their distributions.

## Jupyter Notebook for Interactive Plots in the Paper

Acitivate our virtual python environment "tenuous" and navigate to this folder.

Then open the notebook named "PaperResultIllustrationipynb" and follow the instructions in the notebook. The notebook generates the some interactive plots for assisting users better understanding the paper. 

## Authors

* **John Wilson** - *johnrwilson@uchicago.edu*
* **Jiaming Wang** - *jiamingwang@uchicago.edu*

## Acknowledgments

* Thanks to Lloyd Han and Yiran Fan for a preliminary version of this code in Matlab
