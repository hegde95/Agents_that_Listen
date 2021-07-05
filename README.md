# Agents that Listen: High-Throughput Reinforcement Learning with Multiple Sensory Systems
Train an agent to play VizDoom with multi sensory inputs. Trained using sample factory

**Paper:** ADD ONCE DONE

**Website:** https://sites.google.com/view/sound-rl


Tested on Ubuntu 18.04 64-bit.

- Install miniconda for Python 3.7: https://docs.conda.io/en/latest/miniconda.html

- Clone the repo: `git clone https://github.com/hegde95/ViZDoom_Sound.git`

- Create and activate conda env:

```
cd ViZDoom_Sound
conda env create -f environment.yml
cd ..
conda activate sound
```

This will install sample-factory into the environment too

- Clone the required ViZDoom repo that contains the sound state space: `git clone -b doom_bot_project https://github.com/hegde95/ViZDoom.git`

- Build and install the new environemnt:

```
cd ViZDoom
python setup.py build && python setup.py install
cd ..
```

- Enter the ViZDoom_Sound folder and run the following command to start a training run on the instruction scenario discussed in the paper

```
cd ViZDoom_Sound
python -m rl.train --algo=APPO --env=doomsound_instruction --experiment=doom_instruction --encoder_custom=vizdoomSoundFFT --train_for_env_steps=500000000 --num_workers=24 --num_envs_per_worker=20
```

- To view a rollout of the learned policy run the following line:

```
python -m rl.enjoy --algo=APPO --env=doomsound_instruction --experiment=doom_instruction
```

- To record a demo of the learned policy run the following command:

```
python -m demo.enjoy_appo --algo=APPO --env=doomsound_instruction --experiment=doom_instruction
```
