# Macroeconomic Uncertainty Prices when Beliefs are Tenuous

This repository contains code which estimates the empirical model in "Macroeconomic Uncertainty Prices when Beliefs are Tenuous" by [Lars Peter Hansen][id1] and [Thomas J Sargent][id2]. Appendix B and C from that paper outline the methodology employed in this software.

[id1]: https://larspeterhansen.org/
[id2]: http://www.tomsargent.com/

## Getting Started

To copy the code to your machine, you may either download it from the Github website directly or you may clone the repository in read-only mode.

### Prerequisites

This project simply requires the Anaconda distribution of Python version 3.x. Additional dependencies and prerequisites are handled automatically in setup.

### Installing and activating the environment

Navigate to the folder containing the code and set up the virtual environment necessarily to run our code

For Mac Users, please open the terminal and run the following commands in order
```
cd /path
git clone https://github.com/lphansen/TenuousBeliefs.git
cd TenuousBeliefs
source setup.sh
```
For Windows Users, please open command prompt (shortcut: Windows+R and type 'cmd'）
```
cd /path
git clone https://github.com/lphansen/TenuousBeliefs.git
conda update conda
conda env create -f environment.yml
conda activate tenuous
```
Please replace /path to user designated folder path in both cases.

Press `y` to proceed with installation when prompted. You will know that setup has been correctly implemented if the word `(tenuous)` contained in parenthesis appears on the current line of your terminal window.

## VAR Estimation

### Running the estimation

To run the code, simply use

```
python tenuous_estimation.py
```

This code will print in terminal the estimated 10th, 50th, and 90th percentiles for the data. The results printed as weighted percentiles should be close to the results listed in Appendix B, with variations in random number generation accounting for any differences. The code will also produce a histogram for each of the relevant parameters, showing their distributions.

## Jupyter Notebook for Interactive Plots in the Paper

To run the notebook, simply use: (Makse sure acitivating our virtual python environment "tenuous" and navigating to this folder)
```
jupyter notebook
```

Then open the notebook named "PaperResultIllustrationipynb" and follow the instructions in the notebook. The notebook generates the some interactive plots for assisting users better understanding the paper. 

<a href="https://colab.research.google.com/github/lphansen/TenuousBeliefs/blob/master/PaperResultIllustration.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Uninstalling

Delete the files and open Terminal/Command Prompt and run
```
conda env remove -n tenuous
```

## Authors

* **John Wilson** - *johnrwilson@uchicago.edu*
* **Jiaming Wang** - *jiamingwang@uchicago.edu*
Please feel free to contact us for any types of questions

## Acknowledgments

* Thanks to Lloyd Han and Yiran Fan for a preliminary version of this code in Matlab
