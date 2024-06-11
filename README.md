# kinase_resistance_prediction
Machine learning models to predict resistance of kinase mutants against different cancer drugs (inhibitors)

This repository is a collection of my different attempts at building ML models for predicting the fitness of a given MET kinase sequence variant bound to an inhibitor. I start with the just the mutated kinase sequence and SMILES string and/or molecular weight as input to the models. 

The models are trained on Deep Mutational Scanning experiment data, where the fitness of every possible single point mutant of MET kinase bound to 11 different inhibitors was measured
