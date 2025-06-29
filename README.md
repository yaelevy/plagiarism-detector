# plagiarism-detector

This project is a plagiarism detector, using the arxiv articles database.

The first step of this project is downloading the data and understanding it using simple clustering methods and umap visualisation.
This code can be found in: Plagiarism_Filter_Phase.ipynb

The second step is using the pan-plagiarism database of plagiarism examples and building a model that learns how plagiarism looks like.
The final step is using the model to find plagiarism instances in the arxiv database.

There are additional steps of fine tuning and improving results.
The rest of the code can be found here in the python scripts and sh files.

The correct way to call the plagiarism modul:

run_plagiarism_detection.sh --load_model ./plagiarism_output/best_siamese_bert.pth --test 'src text' 'test text'
