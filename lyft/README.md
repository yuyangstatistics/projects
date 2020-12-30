## Kaggle: Lyft Motion Prediction for Autonomous Vehicles

Check the [competition overview](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview) for more details.

**Team Name**: Statdogs 
**Team members**: Dewei Chen, Xuesong Hou, Chunlin Li, Yu Yang
**Ranking in Leaderboard**: 52nd out of 935 teams.

In this competition, the datasets have already been preprocessed in a good format by the package [l5kit](https://github.com/lyft/l5kit). Our models are CNN-based, and the final model is the ensemble of ResNet18, ResNet34, DenseNet121, EffecientNetB4, and EfficientNetB7.

### Data
Data is accessible in the [Kaggle page](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data).

### Main Code
- `train.py`: train the models.
- `eval.py`: evaluate the trained models on the validation set.
- `pred.py`: predict on the test set.
- `models.py`: the model frameworks using EfficientNet and DenseNet as backbone architectures. (ResNet not included here.)
- `utils.py`: utility functions.