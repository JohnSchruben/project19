[install-dependencies.py]
installs dependencies for the scripts, new models will need different dependencies, add them to this file. 



[runall.py]
runs every py script that accepts a prompt and an image. all the scripts should have optional args and default args for the prompt and image. 

before running runall.py you have to start the models locally. see individual items below.

# model scripts
[ollama-test.py]
# download model https://ollama.com/

# start it
ollama serve

# pull in models
(for llava)
ollama pull llava 

(for minicpm-v: ollama-test.py is using this one but can use llava if you edit the .py file)
pip install ollama pillow
