adl_lstm.py: adl lstm implementation copied from paper

adl_lstm_custom.py: outdated file used for initial testing for a custom solution to the adl dataset

f8.py: initially used for the 1D-CNN model from the paper, as well os other custom adl models, later used to test different models on the f8 dataset, and log their responses to files

f8_no_logging.py: initially used for the 1D-CNN model copied from the paper, as well os other custom adl models, but later adapted to test different model son the f8 dataset, and print reports to the terminal

gridsearch.ipynb: f8_no_logging.py but on JupyterLab (in a notebook, contains most recent runs, not top runs)

helper.py: helper functions including each of the 3 model types, a train, and test function

preprocessing.py: used for creating the block datasets used in the final code