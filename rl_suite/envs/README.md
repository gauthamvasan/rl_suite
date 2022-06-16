# Setup docs


## Compute Canada env setup

```bash
module load python/3.9
virtualenv --no-download rtrl
source rtrl/bin/activate

# Packages 
pip install gym matplotlib numpy
pip install torch torchvision torchaudio

# Mujoco (for linux)
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip -D mujoco200_linux.zip 
mv mujoco200_linux mujoco200
MUJOCO_PATH=/home/vasan/.mujoco/mujoco200 pip install dm_control==0.0.322773188
pip install -U 'mujoco-py<2.2,>=2.1'
pip install mysql-connector-python
```

Add this line to `~/.bashrc`
```bash
# MuJoCo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/vasan/.mujoco/mujoco200/bin
```

