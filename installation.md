# Conda env

N.B: This is on a Macbook Pro. Choose pytorch and other packages accordingly
```
conda create --name rl_suite 
conda install pytorch torchvision torchaudio -c pytorch
pip install gym protobuf numpy matplotlib mujoco_py termcolor tensorboardX
```

## Mujoco setup

```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
tar -xvzf mujoco210-macos-x86_64.tar.gz
mv mujoco210 ~/.mujoco/
```

- If using MacOS, remember to give permission to a .dylib file