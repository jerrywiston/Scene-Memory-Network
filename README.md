# Scene Memory Network
## Requirements
- python = 3.6
- pytorch = 1.10.0
- pyopencv = 4.4.0
- pyrender = 0.1.39
- trimesh = 3.8.18
- pyglm = 1.99.3
- pyopengl = 3.1.0
- matplotlib = 3.3.4

You can create a conda environment :
```
conda env create -f smn.yml
```

## Run Demo
```
python smn_demo.py
```

### Controlling
- <kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd> : Move Forward / Turn Left / Move Backward / Turn Right.
- <kbd>E</kbd> : Add the observation to scene representation.
- <kbd>Space</kbd> : Random teleport to a place in the maze.
- <kbd>Enter</kbd> : Reset maze.
- <kbd>Esc</kbd> : Exit the demo.

## Training
```
# Rendering Model
python train_smn_maze.py --exp_name ${EXP_NAME} --config maze.conf  # SMN
python train_gqn_maze.py --exp_name ${EXP_NAME} --config maze.conf  # GQN

# Reinforcement Learning
python run_dqn.py --exp_name ${EXP_NAME} --n_items 15
```

## Evaluation
```
# Rendering Model
python eval_render.py --path ${EXP_PATH} \
--render_core <smn/strgqn/gqn/gtmsm> \
--eval_type <local/base/global>

# Reinforcement Learning
python run_dqn.py --exp_name ${EXP_NAME} --n_items 15 --test
```
