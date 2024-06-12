# kinase_resistance_prediction
Machine learning models to predict resistance of kinase mutants against different cancer drugs (inhibitors)

This repository is a collection of my different attempts at building ML models for predicting the fitness of a given MET kinase sequence variant bound to an inhibitor. I start with the just the mutated kinase sequence and SMILES string and/or molecular weight as input to the models. 

The models are trained on Deep Mutational Scanning experimental data, where the fitness of every possible single point mutant of MET kinase bound to 11 different inhibitors was measured

The *model_training_scripts* folder contains the python scripts where I have tried out different ML algorithms (linear regression, neural network, random forest, xgboost, etc.)

The *feature_extraction_scripts* folder contains python scripts and sge scripts (bash scripts used for running in UCSF compute clusters) which I used for predicting the structures of kinase mutants bound to inhibitors and for extracting various features from them.

The model that was finally successful is published in the fraserlab github. The repository will be made public once the preprint is uploaded.