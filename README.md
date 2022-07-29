# HILAND

In this repository we provide access to the code used to reproduce the results from our work "Fast  Incremental Learning by Transfer Learning and Hierarchical  Sequencing". 

## Dependencies
Our experiments can be reproduced in any operating system that supports the following dependencies:
- Python 3.6.9
- CUDA 11.0 (optional, only for faster training/testing times)

## Installation
1. Download our code repository:
   
   `git clone https://github.com/capo-urjc/HILAND.git`
   
2. Move into the cloned project:
   
   `cd HILAND`

3. Create a python virtual environment:
   
   `python -m venv venv`
   
3. Activate the newly created virtual environment:
   - Linux:
      
      `source venv/bin/activate`
   
   - Windows:
      
      `venv\Scripts\activate.bat`
   
4. Upgrade `pip` to prevent errors while installing newer packages:
   
   `pip install --upgrade pip`
   
5. Install our basic requirements:
   
   `pip install -r requirements.txt`

6. Install the required PyTorch version:
   - If using a CUDA 11.0 enabled GPU:
      
      `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
   
   - Otherwise, install the CPU version (results may differ):
   
      `pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

## Usage
To run our experiments we provide 20 configuration files in [exp1_5/exp_config](exp1_5/exp_config) and [exp6_7/exp_config](exp6_7/exp_config).
To run our experiments use the following commands:

1. From a terminal window in the project root directory, activate the virtual environment created during the installation process:
   - Linux:
      
      `source venv/bin/activate`
   
   - Windows:
      
      `venv\Scripts\activate.bat`

2. Run our experiments:
   - Linux:
      
      `./run_experiments.sh` 
     (use `chmod +x run_experiments.sh` if you encounter permission issues when running the previous command)
   
   - Windows:
      
      `run_experiments.bat`

3. Check the results in the CSV files in the newly created  `exp_out` folder.




