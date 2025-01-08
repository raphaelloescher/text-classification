# Text Classification for Harware Data with BERT
## Training
The training.ipynb is the jupyter notebook used for training and preprossesing. Installation instruction:

Create a new Kernel for your jupyter lab with the requirements.txt installed. Afterwards run the file (best case with GPU).
Step by step
```
pip install -r requirements.txt
```
```
python -m venv text_classification_env
```
```
source text_classification_env/bin/activate
```
```
pip install ipykernel
```
```
python -m ipykernel install --user --name=bert_env --display-name "Python (Text Classification Env)"
```
```
jupyter notebook
```
(Launch with the new kernel)
## Vue/Flask Demo
For a running application just run the app.py in the flask-vue-demo folder.

## Linux
- Git
```
sudo yum update
sudo yum install git
```
- VSCode
```
sudo yum install code
```
```
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" | sudo tee /etc/yum.repos.d/vscode.repo > /dev/null
```
(Pyhton and Vue extension)