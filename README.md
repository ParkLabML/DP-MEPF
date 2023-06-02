# Differentially Private Mean Embeddings with Perceptual Features (DP-MEPF) 
  
# Disclaimer: Error in FID eval - subject to change
### The error is due to the normalization of data excuted before passing them through the inception network. The correct FID scores are likely to be higher than what was presented in the paper. 
### This affects tables 3, 4 and 5 in the main text and tables 12, 13, 14, 18, and 19 in the supplementary material.
### We are working on correcting this error and will upload a new version of the code along with updated results whithin the next few weeks. Please do not rely on the provided FID evaluation while this disclaimer is up.



This is the code for the paper **Pre-trained Perceptual Features Improve Differentially Private Image Generation** by Frederik Harder, Milad Jalali, Danica J. Sutherland and Mijung Park, published in TMLR (https://openreview.net/forum?id=R6W7zkMz0P).



The code is based on the implementation for Generative Feature Matching Networks (https://github.com/IBM/gfmn), adapted for python 3 and Pytorch 1.10.


### Repository Structure
- `code/` contains our implementation of dp-mepf as used in the paper.
  - refer to `code/README.md` for instruction on how to run things.
- `data/` is the default path for datasets
- `logs/` is the default path for experiment logs
- `requirements.txt` lists the required packages
