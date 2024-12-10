# Scene GNNs

6.7960
Christy, Claire, Gracie

## Generating Data

Atari: 
```
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2
```

Space Invaders
```
python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_train.h5 --num_episodes 1000 --atari --seed 1
python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_eval.h5 --num_episodes 100 --atari --seed 2
```

3 Body Gravitational Physics: 
```
python data_gen/physics.py --num-episodes 5000 --fname data/balls_train.h5 --seed 1
python data_gen/physics.py --num-episodes 1000 --fname data/balls_eval.h5 --eval --seed 2
```

## Training and Eval

Atari: 
```
python train.py --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong
python eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/pong --num-steps 1
```

Space Invaders:
```
python train.py --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name spaceinvaders
python eval.py --dataset data/spaceinvaders_eval.h5 --save-folder checkpoints/spaceinvaders --num-steps 1
```

3 Body: 
```
python train.py --dataset data/balls_train.h5 --encoder medium --embedding-dim 4 --num-objects 3 --ignore-action --name balls
python eval.py --dataset data/balls_eval.h5 --save-folder checkpoints/balls --num-steps 1
```


## Transfer Learning 

Train a model on one game and then transfer it to another game, or on the physics simulations. 

```
python transfer.py \
    --pretrained-model checkpoints/pong_k3/model.pt \
    --new-dataset data/spaceinvaders_train.h5 \
    --batch-size 512 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --encoder medium \
    --action-dim 6 --name space_transfer_100 \
    --decoder \
    --device-id 0
```

## Latent Visualizations 

```
python latent_vis.py --model-path checkpoints/spaceinvaders/model.pt \
	--meta-path checkpoints/spaceinvaders/metadata.pkl \
	--dataset data/spaceinvaders_eval.h5 \
	--save-dir spaceinvaders_visualizations
```