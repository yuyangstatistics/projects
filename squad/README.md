# Retro-Reader
This is a project for the STAT 8931 course.
- `data` contains the raw data.
- `save` contains the model training and testing logs.
- `src` contains the python scripts and shell scripts.
- `report.pdf` is the report for this project.
- `slides.pdf` are the presentation slides.
- `presentation.mp4` is the presentation video.
- `environment.yml` is the configuration file to set up the conda virtual environment.
- `LICENSE` is the license file of the CS224n course code framework.

## References
1. The [skeleton of the code framework](https://github.com/minggg/squad) is adapted from the [CS224n default project assignment](https://web.stanford.edu/class/cs224n/project/default-final-project-handout.pdf) (Used under the license.)
2. The idea of retrospective reader is from the paper [*Retrospective Reader for Machine Reading Comprehension*](https://arxiv.org/abs/2001.09694) by Zhuosheng Zhang, Junjie Yang, and Hai Zhao.

## Set Up
1. Create a conda environment using the following command in your shell.
    ```bash
    conda env create -f environment.yml
    ```

2. Activate or deactivate the virtual environment using the following commands.
    ```
    conda activate squad
    conda deactivate
    ```
3. Run `python setup.py`.
   - This downloads GloVe 300-dimensional word vectors and the SQuAD 2.0 train/dev sets. This also pre-processes the dataset for efficient data loading. 

## Implementation Details
Change directory to `src` folder: `cd src`.
- `bash sh_bidaf.sh`: train the baseline BIDAF model. The trained model is saved in `save/train/baseline-02/best.pth.tar`.
- `bash sh_sketchy.sh`: train the sketchy reader. The trained model is saved in `save/train/sreader-01/best.pth.tar`.
- `bash sh_intensive.sh`: train the intensive reader. The trained model is saved in `save/train/ireader-05/best.pth.tar`.
- `bash sh_tune_tav.sh`: use grid search to find the optimal threshold value in Threshold-Based Answerable Verification. The tuned threshold is -0.006. The result is saved in `save/test/tune-01/`.
- `bash sh_test_bidaf.sh`: evaluate the trained baseline model on the dev set. The result is saved in `save/test/baseline-01/`.
- `bash sh_tav.sh`: use the tuned threshold to do TAV, and evaluate the retro-reader on the dev set. The result is saved in `save/test/retro_reader-01/`.



## Notice
Due to Github data file size limitations, trained model weights are not committed, but log files should be able to provide some details.