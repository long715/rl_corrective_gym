FROM continuumio/anaconda3

# setup task dependencies 
RUN conda create --name corrective-mars numpy pandas gymnasium &&\
    conda activate corrective-mars &&\
    conda config --add channels conda-forge &&\
    conda config --set channel_priority strict &&\
    conda install pykep &&\ 
    pip install stable-baselines3 &&\ 
    conda install python=3.11

WORKDIR /space-task

# setup cares packages
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git &&\ 
    cd cares_reinforcement_learning &&\
    conda install setuptools==79.0.0 pytest==7.4.2 matplotlib==3.10.1 pandas==1.5.3 PyYAML==6.0.1 &&\ 
    pip install torch==2.7.0 opencv-contrib-python==4.6.0.66 scikit_image==0.25.2 &&\
    pip install --editable . &&\
    cd ..
    
RUN git clone https://github.com/UoA-CARES/gymnasium_envrionments.git &&\
    conda install mujoco==3.2.6 mediapy==1.1.9 natsort==8.4.0 &&\ 
    pip install dm_control==1.0.26  &&\
    git checkout space-gym

'''
    once this docker container is setup in the server; creative an interactive instance (no automatic termination):   
        docker run -it space-image /bin/bash
    access the interactive shell: 
        docker exec -it space-image /bin/bash
    close the docker container: 
        exit

    ----------------
    once docker container is setup, set up github credentials to git clone the gym repository. 
    run the task with the following command: 
        cd gymnasium_envrionments/scripts && \
        python run.py train config --data_path ../../rl-corrective-gym/space_configs/
'''