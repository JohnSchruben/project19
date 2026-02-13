# ollama is the program that interacts with the models
# download 
https://ollama.com/

or on linux cli 
curl -fsSL https://ollama.com/install.sh | sh

# start it, it should start when the pc starts
ollama serve

# pull in models
python .\install_dependencies.py

# run all 
python .\runall.py

# add new models to 
models.py

# find new models here 
https://ollama.com/search