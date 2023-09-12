# RaSP: Relation-aware Semantic Prior for Weakly Supervised Incremental Segmentation [[Paper](https://arxiv.org/abs/2305.19879)]

## Versions:
    - ws_ciss-v1.0
        - Replay old images for the prior loss 
        - Added visualizations
    - ws_ciss-v1.1
        - Replay Imagenet
        - Replay VOC with fixed memory
        - Semantic Similarity loss using sentence transformers
        - Runs both VOC and COCO-to-VOC
    - ws_ciss-v1.2
        - Single stage FSS on VOC
        - some refactoring

## How to run
### Requirements
Code has been tested with the following versions:
```
python == 3.7
pytorch == 1.11
```

If you want to install a custom environment for this code, you can run the following using [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html):
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
conda install tensorboard
conda install jupyter
conda install matplotlib
conda install tqdm
conda install imageio

pip install inplace-abn # this should be done using CUDA compiler (same version as pytorch)
pip install wandb # to use the WandB logger
```

Installing inplace-abn can be tricky. So follow the steps related to inplace-abn in the [file](install.md)

The full list of dependencies have been provided in the requirements.txt file

### Datasets 
In the benchmark there are two datasets: Pascal-VOC 2012 and COCO (object only).
For the COCO dataset, we followed the COCO-stuff splits and annotations, that you can see [here](https://github.com/nightrome/cocostuff/).

To download dataset, follow the scripts: `data/download_voc.sh`, `data/download_coco.sh` 

To use the annotations of COCO-Stuff in our setting, you should preprocess it by running the provided script. \
Please, remember to change the path in the script before launching it!
`python data/coco/make_annotation.py`

The above python script will generate `annotations_my/` folder. The annotations of that folder should be used in the experiments.

If your datasets are in a different folder, make a soft-link from the target dataset to the data folder. For e.g.: 

```ln -s /some/path/coco/annotations_my data/coco/annotations```

We expect the following tree inside the project folder:
```
data/voc/
    SegmentationClassAug/
        <Image-ID>.png
    JPEGImages/
        <Image-ID>.jpg
    split/
    
data/coco/
    annotations/
        train2017/
            <Image-ID>.png
        val2017/
            <Image-ID>.png
    images/
        train2017/
            <Image-ID>.jpg
        val2017/
            <Image-ID>.jpg
```

Finally, to prepare the COCO-to-VOC setting we need to map the VOC labels into COCO. Do that by running
`python make_cocovoc.py`


### ImageNet Pretrained Models
After setting the dataset, you download the models pretrained on ImageNet using [InPlaceABN](https://github.com/mapillary/inplace_abn).
Download the [ResNet-101](https://drive.google.com/file/d/1oFVSIUYAxa_uNDq2OLkbhyiFmKwnYzpt/view) for VOC and [WideResNet-38](https://drive.google.com/file/d/1Y0McSz9InDSxMEcBylAbCv1gvyeaz8Ij/view) for COCO-to-VOC.
Then, put the pretrained model in the `pretrained` folder. Make sure they have the name as:

```
pretrained/
    resnet101_iabn_sync.pth.tar
    wide_resnet38_ipabn_lr_256.pth.tar
```

### Run!
We provide the scripts to run the experiments reported in the paper. The scripts are arranged as:
```
run/
    ablations/
    eval/
    run-voc-rasp.sh
    run-coco-rasp.sh
        .
        .
```
Follow the instructions inside each bash script to launch experiments. Some examples:
1. Run RaSP for 15-5 VOC single-step disjoint incremental setting. `bash run/run-voc-rasp.sh 0,1 0 15-5 RaSP 1`
2. Run RaSP for 10-2 VOC multi-step overlap incremental setting. `bash run/run-voc-rasp.sh 0,1 1 10-2 RaSP 5`

Please remember to change the path to the datasets as per your workspace.

### Acknowledgements
This codebase has been built upon the works [WILSON](https://github.com/fcdl94/WILSON) and [MiB](https://github.com/fcdl94/MiB)


### Reference

**RaSP: Relation-aware Semantic Prior for Weakly Supervised Incremental Segmentation** 

Subhankar Roy, Riccardo Volpi, Gabriela Csurka and Diane Larlus 
```
    @InProceedings{Roy_2023_CoLLAs,
    author = {Subhankar, Roy and Volpi, Riccardo and Csurka, Gabriela and Larlus, Diane},
    title = {RaSP: Relation-aware Semantic Prior for Weakly Supervised Incremental Segmentation},
    booktitle = {The 2nd Conference on Lifelong Learning Agents (CoLLAs)},
    month = {August},
    year = {2023}
    }
```
