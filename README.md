# ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation

This code relates to the publication ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation by Snigdha Sen et al.

A preprint is available at: https://arxiv.org/abs/2309.06268

This code fits the Vascular, Extracellular and Restricted DIffusion for Cytometry in Tumours (VERDICT)-MRI model for prostate using self-supervised deep learning, adapted from https://github.com/sebbarb/deep_ivim.

There are four files included: main.py, verdict_data.py, train.py and base_model.py. Edit main.py and verdict_data.py to contain the corect file path to load the data you want to fit the VERDICT model to, as well as the mask you want to use. base_model.py contains the formulation of the complex VERDICT signal compartments in differentiable form. train.py contains the training protocol. Run main.py to fit the model to your data.

If you have any questions, contact me at snigdha.sen.20@ucl.ac.uk
