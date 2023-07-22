# Differentially Private Mean Embeddings with Perceptual Features (DP-MEPF) 
  
This is the code for the paper **Pre-trained Perceptual Features Improve Differentially Private Image Generation** by Frederik Harder, Milad Jalali, Danica J. Sutherland and Mijung Park, published in TMLR (https://openreview.net/forum?id=R6W7zkMz0P).

The code is based on the implementation for Generative Feature Matching Networks (https://github.com/IBM/gfmn), adapted for python 3 and Pytorch 1.10.

### Previous disclaimer:
Note that our previous code (https://github.com/ParkLabML/DP-MEPF/tree/main/code/old_code) had an error in FID computation due to a wrong scaling of data. We fixed this issue on July 20, 2023, and updated our code and the paper on both ArXiv (https://arxiv.org/pdf/2205.12900.pdf) and TMLR accordingly. 


### Repository Structure
- `code/` contains our implementation of dp-mepf as used in the paper.
  - refer to `code/README.md` for instruction on how to run things.
- `data/` is the default path for datasets
- `logs/` is the default path for experiment logs
- `requirements.txt` lists the required packages
