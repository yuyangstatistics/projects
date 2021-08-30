
This is the code for my STAT 8056 course project. The model is built on top of the [Pointer-Generator model](https://arxiv.org/abs/1704.04368) and the implementation for the PGNet model is adapted from the code in [atulkum/pointer_summarizer](https://github.com/atulkum/pointer_summarizer).

- [Experiment Results](#experiment-results)
  - [Train the original pointer-generator model (with coverage loss disabled)](#train-the-original-pointer-generator-model-with-coverage-loss-disabled)
  - [Train the proposed model (PGNet + Text RBM)](#train-the-proposed-model-pgnet--text-rbm)
- [Environment Configurations](#environment-configurations)
- [Training Details](#training-details)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Model Training](#model-training)


## Experiment Results
### Train the original pointer-generator model (with coverage loss disabled)
After training for 100k iterations (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3698 with confidence interval (0.3674, 0.3721)
rouge_1_recall: 0.3907 with confidence interval (0.3880, 0.3933)
rouge_1_precision: 0.3758 with confidence interval (0.3730, 0.3786)

ROUGE-2:
rouge_2_f_score: 0.1593 with confidence interval (0.1572, 0.1615)
rouge_2_recall: 0.1675 with confidence interval (0.1652, 0.1699)
rouge_2_precision: 0.1631 with confidence interval (0.1607, 0.1655)

ROUGE-l:
rouge_l_f_score: 0.3355 with confidence interval (0.3332, 0.3377)
rouge_l_recall: 0.3541 with confidence interval (0.3517, 0.3567)
rouge_l_precision: 0.3412 with confidence interval (0.3386, 0.3439)
```

After training for 500k iterations (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3592 with confidence interval (0.3570, 0.3616)
rouge_1_recall: 0.3607 with confidence interval (0.3584, 0.3634)
rouge_1_precision: 0.3812 with confidence interval (0.3785, 0.3841)

ROUGE-2:
rouge_2_f_score: 0.1551 with confidence interval (0.1531, 0.1573)
rouge_2_recall: 0.1552 with confidence interval (0.1530, 0.1575)
rouge_2_precision: 0.1658 with confidence interval (0.1633, 0.1682)

ROUGE-l:
rouge_l_f_score: 0.3287 with confidence interval (0.3265, 0.3311)
rouge_l_recall: 0.3299 with confidence interval (0.3276, 0.3325)
rouge_l_precision: 0.3492 with confidence interval (0.3465, 0.3519)
```

![Alt text](learning_curve_pg.png?raw=true "Learning Curve with original pgnet")

### Train the proposed model (PGNet + Text RBM)
After training for 100k iterations (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3633 with confidence interval (0.3610, 0.3656)
rouge_1_recall: 0.3843 with confidence interval (0.3818, 0.3869)
rouge_1_precision: 0.3690 with confidence interval (0.3660, 0.3717)

ROUGE-2:
rouge_2_f_score: 0.1562 with confidence interval (0.1541, 0.1584)
rouge_2_recall: 0.1648 with confidence interval (0.1625, 0.1671)
rouge_2_precision: 0.1596 with confidence interval (0.1572, 0.1621)

ROUGE-l:
rouge_l_f_score: 0.3305 with confidence interval (0.3282, 0.3328)
rouge_l_recall: 0.3494 with confidence interval (0.3469, 0.3519)
rouge_l_precision: 0.3359 with confidence interval (0.3332, 0.3386)
```

After training for 500k iterations (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3633 with confidence interval (0.3608, 0.3657)
rouge_1_recall: 0.3631 with confidence interval (0.3604, 0.3656)
rouge_1_precision: 0.3880 with confidence interval (0.3851, 0.3909)

ROUGE-2:
rouge_2_f_score: 0.1583 with confidence interval (0.1560, 0.1607)
rouge_2_recall: 0.1574 with confidence interval (0.1550, 0.1599)
rouge_2_precision: 0.1705 with confidence interval (0.1679, 0.1733)

ROUGE-l:
rouge_l_f_score: 0.3325 with confidence interval (0.3300, 0.3348)
rouge_l_recall: 0.3321 with confidence interval (0.3294, 0.3347)
rouge_l_precision: 0.3554 with confidence interval (0.3525, 0.3582)
```
![Alt text](learning_curve_rbmpg.png?raw=true "Learning Curve with rbmpg")


## Environment Configurations
1. Set up the virtual environment.
    ```bash
    conda env create -f environment.yml
    ```
1. The implementation in [atulkum/pointer_summarizer](https://github.com/atulkum/pointer_summarizer) uses pytorch 0.4 and python 2.7, for the sake of convenience, we use the same version. Since torch 0.4 is not supported on CUDA 10.0, so we resort to CUDA 9.2. Since Tensorflow-gpu is only compatible with 9, not 9.2, so install tensorflow with cpu only.
2. Install tensorflow
	```bash
	pip install --upgrade tensorflow==1.15.0
	```
4. Install pytorch 0.4 to be compatible with the code.
	```bash
	conda search cudatoolkit
	conda search cudnn
	conda install cudatoolkit=9.2
	conda install cudnn=7.6.0=cuda9.2_0
	module load cuda/9.2.148
	conda install pytorch=0.4.1 cuda92 -c pytorch
	```
4. pip install the **nltk** package.
	```bash
	pip install nltk==3.4.5
	```
5. Install **pyrouge** and **ROUGE** without root.

	Reference: 
    - [stackoverflow: installing pyrouge and ROUGE in ubuntu](https://stackoverflow.com/a/57686103/13448382)
	- [stackoverflow: use CPAN as a non-root user](https://stackoverflow.com/questions/2980297/how-can-i-use-cpan-as-a-non-root-user)
	- [github issues: install XML::Parser](https://github.com/pltrdy/files2rouge/issues/9#issuecomment-593850124)
	```bash
	# step 1: install pyrouge from source
	git clone https://github.com/bheinzerling/pyrouge
	cd pyrouge
	pip install -e .
	# step 2: install official ROUGE script
	git clone https://github.com/andersjo/pyrouge.git rouge
	# step 3: point pyrouge to official rouge script
	pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/
	# step 4: install xml parser
	wget -O- http://cpanmin.us | perl - -l ~/perl5 App::cpanminus local::lib
	eval `perl -I ~/perl5/lib/perl5 -Mlocal::lib`
	echo 'eval `perl -I ~/perl5/lib/perl5 -Mlocal::lib`' >> ~/.bash_profile
	echo 'export MANPATH=$HOME/perl5/man:$MANPATH' >> ~/.bash_profile
	# step 5: regenerate the Exceptions DB
	cd rouge/tools/ROUGE-1.5.5/data
	rm WordNet-2.0.exc.db
	./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
	# step 6: run the tests
	python -m pyrouge.test
	```
	

## Training Details

### Data Loading and Preprocessing
1. Follow data generation instruction from https://github.com/abisee/cnn-dailymail.
   1. We need to use [corenlp-3.7.0](https://stanfordnlp.github.io/CoreNLP/history.html) for the script to work.
   2. We need to load java `module load jdk/9.0.1` in the dags server.
2. Run `python train_textRBM/get_top15k` to generate a text file containing the 15k most frequent words in the training dataset.

### Model Training
1. Change the paths and parameters in `data_util/config.py`.
2. Run `bash start_rbm.sh` to train the Replicated Softmax model.
3. Run `bash start_train.sh` to train the Pointer-Generator model.
4. Run `bash start_decode.sh` to decode.
5. Run `bash start_eval.sh` to evaluate using ROUGE.

Note:

* In decode mode beam search batch should have only one example replicated to batch size
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/decode.py#L109
https://github.com/atulkum/pointer_summarizer/blob/master/data_util/batcher.py#L226
* All the log files have been removed from the repository due to privacy concern.

