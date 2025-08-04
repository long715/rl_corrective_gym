# Installation 
- install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), `miniforge` should be sufficient 
- create a conda environment and install the following packages `conda create --name <my-env> numpy pandas gymnasium stable-baselines3`
- activate the conda environment `conda activate <my-env>` 
- install [pykep](https://esa.github.io/pykep/installation.html)
- install stable-baselines3 with pip (currently unstable in conda-forge) `pip install stable-baselines3`

notes: 
- need to be python v3.11 for cares-rl `conda install python=3.11`
- have to `pip install torch==2.7.0 opencv-contrib-python==4.6.0.66 scikit_image==0.25.2`
- need to pip install some packages for gymnasium_env, `pip install dm_control==1.0.26`

# Run
`python run.py train config --data_path ../../rl_corrective_gym/rl_corrective_gym/space_configs`
- need to use / in MacOS
- mps not supported, need to use cpu in MacOS

# Using the Docker container
- transfer Dockerfile to server `scp Dockerfile user@ip:/env_dir/`; `scp source target`
- build the Dockerfile `docker buildx build -t space-image .`
- run the container as a background task `docker run -d --gpus all -v ~/cares_rl_logs:/root/cares_rl_logs space-image`, this also mounts the results to the cares_log in the server
- open the container as an interactive shell `docker exec -it <id> bash`

# Documentation 
- replace the parser with custom env config class
- change the function name for obs space shape