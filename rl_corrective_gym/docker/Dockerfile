FROM continuumio/anaconda3

# setup task dependencies 
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh &&\
    conda create --name corrective-mars numpy pandas gymnasium &&\
    conda activate corrective-mars &&\
    conda config --add channels conda-forge &&\
    conda config --set channel_priority strict &&\
    conda install pykep &&\ 
    pip install stable-baselines3 &&\ 
    conda install python=3.11"

WORKDIR /space-task

# setup cares packages, need to activate conda env for each bash session

RUN bash -c "git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git &&\ 
    cd cares_reinforcement_learning &&\
    source /opt/conda/etc/profile.d/conda.sh &&\
    conda activate  corrective-mars &&\
    conda install setuptools==79.0.0 pytest==7.4.2 matplotlib==3.10.1 pandas==1.5.3 PyYAML==6.0.1 seaborn==0.13.0 numpy==1.26.4 pydantic==1.10.13 &&\ 
    pip install torch==2.7.0 opencv-contrib-python==4.6.0.66 scikit_image==0.25.2 &&\
    pip install --editable . &&\
    cd .."
    
RUN bash -c "git clone https://github.com/UoA-CARES/gymnasium_envrionments.git &&\
    source /opt/conda/etc/profile.d/conda.sh &&\
    conda activate  corrective-mars &&\
    conda install mujoco==3.2.6 mediapy==1.1.9 natsort==8.4.0 &&\ 
    pip install dm_control==1.0.26  &&\
    cd gymnasium_envrionments &&\
    git checkout space-gym &&\
    cd .."

RUN bash -c "git clone https://github.com/long715/rl_corrective_gym.git &&\ 
    cd rl_corrective_gym &&\
    source /opt/conda/etc/profile.d/conda.sh &&\
    conda activate  corrective-mars &&\
    pip install --editable . &&\
    cd ../gymnasium_envrionments/scripts &&\
    python run.py train config --data_path ../../rl_corrective_gym/rl_corrective_gym/space_configs"    

# run the following after building: docker run -it --gpus all -v ~/cares_rl_logs:/root/cares_rl_logs space-image bash