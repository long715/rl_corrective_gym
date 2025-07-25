# Installation 
- install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), `miniforge` should be sufficient 
- create a conda environment and install the following packages `conda create --name <my-env> numpy pandas gymnasium stable-baselines3`
- activate the conda environment `conda activate <my-env>` 
- install [pykep](https://esa.github.io/pykep/installation.html)
- install stable-baselines3 with pip (currently unstable in conda-forge) `pip install stable-baselines3`

notes: 
- need to be python v3.11 for cares-rl, have to `pip install torch==2.7.0 opencv-contrib-python==4.6.0.66 scikit_image==0.25.2`
- need to pip install some packages for gymnasium_env, `pip install dm_control==1.0.26`

# Run
`python run.py train config --data_path ..\..\space_configs\`