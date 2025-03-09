# Fed-Avg-Base

> This repository contains the boiler plate code for Synchronous Federated Learning Research. Please feel free to use this code for your research to reduce development time for your experiments. There are perhaps too many repositories for Federated Learning Simulations but it's often hard to use for beginners  and the design is not generic enough. This repository can be used to implement new aggregation strategies, new loss functions, new training techniques and FL attacks. 


# Environment setup
Install Miniconda for a stable python version from `https://repo.anaconda.com/miniconda/`. In the future, other versions can be used. 

Enter your `base` conda environment and run the following comands to create your environment. 
```bash
conda create -n fl python=3.10 pip jupyter -y
conda activate fl
pip install -r requirements.txt
```


