# CoDiS
Implementation on Paper **Conditional Diffusion for Causal Inference with State Space Representation**


## Introduction

Causal inference from observational data, particularly for estimating individual-level effects, is vital in high-stakes domains like healthcare and economics, yet remains challenging. A key difﬁculty lies in modeling complex interdependencies, such as temporal patterns in patient histories or structural relationships among economic indicators, which are often overlooked by methods relying on unstructured covariate inputs. Furthermore, most existing approaches only provide point estimates for potential outcomes, neglecting the crucial distributional information needed to quantify uncertainty and assess individual-level risks. To address these gaps, this paper introduces CoDiS, a novel framework that integrates the strengths of state space model (SSM) and conditional diffusion processes for causal inference. Speciﬁcally, CoDiS leverages the inherent inductive biases of SSMs to explicitly capture complex dependencies within covariates and treatment, learning structured, dependency-aware representations. Conditioned on these informative representations, a conditional diffusion process then generates the full potential outcome distributions, moving beyond point estimates to enable reliable uncertainty quantiﬁcation. Comprehensive experiments demonstrate that CoDiS achieves state-of-the-art or comparable performance against leading baselines in both point estimation accuracy (PEHE, RMSE) and distributional ﬁdelity (Wasserstein distance, PI calibration) across diverse benchmark datasets.

## Setup
### Installation:
`python 3.8.18 
pytorch 1.12.1
numpy 1.24.3`


### Getting started:


#### Prerequisites:
Before running the experiments, download datasets [IHDP dataset](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets),
[ACIC2016](https://jenniferhill7.wixsite.com/acic-2016/competition), 
[ACIC2018](https://www.synapse.org/Synapse:syn11294478/wiki/486304)
 and preprocessing them. 


Organize the datasets into their respective folders (`dataset_mask` and `dataset_norm_data`), following the example below.

#### Training example:


The original downloaded data (ACIC2018 dataset) are preprocessed using `load_acic2018.ipynb` and stored in the `acic2018_mask` and `acic2018_norm_data` folders.
```
data/
├── acic2018/
    ├── acic2018_norm_data/
    └── acic2018_mask/
```
The default hyperparameters are set in `./config/acic2018.yaml`. 

An example of running CoDiS is given by `./script/script_acic2018.sh`.


