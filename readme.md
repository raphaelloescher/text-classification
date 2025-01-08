# Text Classification for Harware Data
## Training
The training.ipynb is the jupyter notebook used for training and preprossesing. Installation instruction:
- Create a new Kernel for your jupyter lab with the requirements.txt installed. Afterwards run the file (best case with GPU).
Step by step
- pip install -r requirements.txt
- python -m venv text_classification_env
- source bert_env/bin/activate
- pip install ipykernel
- python -m ipykernel install --user --name=bert_env --display-name "Python (Text Classification Env)"
- jupyter notebook (Launch with the new kernel)
## Vue/Flask Demo
For a running application just run the app.py in the flask-vue-demo folder.