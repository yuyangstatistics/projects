# Kaggle: Lyft Motion Prediction for Autonomous Vehicles


**Team Name**: Statdogs

**Team members**: Dewei Chen, Xuesong Hou, Chunlin Li, Yu Yang

**Ranking in Leaderboard**: [52nd](https://www.kaggle.com/yuyangstat) out of 935 teams.

The [datasets](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data) for this competition are well preprocessed by the package [l5kit](https://github.com/lyft/l5kit). Check the [competition overview](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview) for more details.

### Main Code
 Our models are CNN-based, and the final model is the ensemble of ResNet18, ResNet34, DenseNet121, EffecientNetB4, and EfficientNetB7.
- `train.py`: train the models.
- `eval.py`: evaluate the trained models on the validation set.
- `pred.py`: predict on the test set.
- `models.py`: the model frameworks using EfficientNet and DenseNet as backbone architectures. (ResNet not included here.)
- `utils.py`: utility functions.