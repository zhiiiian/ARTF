

# ARTF

The code of the paper "Adversarially Regularized Tri-Transformer Fusion for Continual Multimodal Egocentric Activity Recognition".



## **How to use**

### Clone

```
git clone https://github.com/zhiiiian/ARTF.git
cd ARTF
```

### Environment Configuration

Firstly, you should make a new environment with python>=3.6, for example:

```
conda create -n artf python=3.8
```

Next, you can download pytorch from official site, for example:

```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, run `pip install -r requirements.txt` in this repo to install a few more packages.

Lastly, install [Timesformer](https://github.com/facebookresearch/TimeSformer?tab=readme-ov-file) according to its official guide.

### Dataset Preparation

We evaluate our model on [UESTC-MMEA-CL](https://ivipclab.github.io/publication_uestc-mmea-cl/mmea-cl/), which is the first multimodal dataset for continual egocentric activity recognition. You can download this dataset from its homepage and put them under the holder `dataset`. The file structure would look like:

```
ARTF/
|-- backbone/
|-- convs/
|-- dataset/
|   |-- video/
|   |   |-- 1_upstairs/
|   |   |-- ...
|   |   |-- 32_watch_TV/
|   |-- mpu/
|   |   |-- 1_upstairs/
|   |   |-- ...
|   |   |-- 32_watch_TV/
|-- exps/
|-- models/
|-- ops/
|-- TimeSformer/
|-- utils/
|-- video_records/
|-- main.py
|-- model_T.py
|-- mydataset_train.txt
|-- mydataset_test.txt
|-- opts.py
|-- requirements.txt
|-- trainer.py
|-- transforms.py
```

### Train

If you want to train and test ARTF with different modalities, you can run the following example:
```
bash train.sh
```

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [PyCIL](https://github.com/G-U-N/PyCIL/tree/master)
- [TBN](https://github.com/ekazakos/temporal-binding-network/tree/master)
- [Timesformer](https://github.com/facebookresearch/TimeSformer?tab=readme-ov-file)
