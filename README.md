# Advancing Collaborative Debates with Role Differentiation through Multi-Agent Reinforcement Learning

### [Project Page](https://anonymous.4open.science/r/MARDan-2513/) 


This is a preliminary implementation of the paper "Advancing Collaborative Debates with Role Differentiation through Multi-Agent Reinforcement Learning". More tasks and settings will be released soon. 

## Running experiments

The code for running AddSub, SingleEQ, MultiArith, GSM8k, ASDiv, SVAMP, and MATH tasks may be found in the following subfolders

* ./src/qmix_extradata_addsub/ contains code for running AddSub
* ./src/qmix_extradata_singleeq/ contains code for running SingleEQ
* ./src/qmix_extradata_multiarith/ contains code for running MultiArith
* ./src/qmix_extradata_gsm/ contains code for running GSM8k
* ./src/qmix_extradata/ contains code for running ASDiv
* ./src/qmix_extradata_svamp/ contains code for running SVAMP
* ./src/qmix_extradata_math/ contains code for running MATH.

## Installation instructions
﻿
Install Python packages
﻿
```shell
# require Anaconda 3
conda create -n mard python=3.10.13
conda activate mard
```
To set up your environment for the project, ensure you have the following dependencies installed:

```bash
accelerate==0.29.2
datasets==2.18.0
tqdm==4.66.2
transformers==4.44.0
peft==0.9.0
torch==2.4.0 
trl==0.8.6
deepspeed==0.14.4
```
