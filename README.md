# SMILES-Reinforcement-Learning-Model
This repository implements a SMILES-based reinforcement learning framework to quantify visual complexity of organic molecules. It extracts molecular descriptors via RDKit, trains an RL agent to rank diagrams by complexity using human feedback, and evaluates alignment via Levenshtein distance.

Abstract: We present a novel computational framework for quantifying the visual complexity of organic chemical structures and for predicting the cognitive load they impose during learning. Our approach converts SMILES representations into normalized feature vectors capturing atom counts, bond types, ring descriptors, stereochemical notation, and formal charge. These vectors serve as inputs to a reinforcement-learning agent which develops a policy, which is the weighting and calculations of the features, and based on the single numerical output, assigns each molecule a normalized complexity score from 0-100, with 0 being the least complex and 100 being the most complex. After separating the diagrams into quintiles, the agent learns via a reward derived from Levenshtein distance comparisons to pilot human rankings to produce consistent and human‐aligned complexity orderings. Our results demonstrate that SMILES-based feature engineering combined with reinforcement learning can approximate human perceptions of diagram complexity with some degree of accuracy. Future work will expand human sorting studies, integrate eye-tracking metrics, and explore deep-learning embeddings to capture spatial layout more fully.

Please see our paper at: https://docs.google.com/document/d/1VtuPxiVDN5ev3vOd_i0OzNbVQaa89prtpVSsB_FPvW4/edit?usp=sharing
All have permission to comment. Please freel free to ask questions or suggest improvements. Thank you!
-Tommaso R. Marena, Catholic University of America, Chemistry Department

A special thanks Dr. Katherine Havanki for being an wonderful research mentor who never ceases to inspire curiosity. 
