# Semantic Information Oriented No-Reference Video Quality Assessment

## Description
SIONR code for the following papers:

- Wei Wu, Qinyao Li, Zhenzhong Chen, Shan Liu. Semantic Information Oriented No-Reference Video Quality Assessment.
![Framework](Framework.png)
## Feature Extraction
```
python generate_CNNfeatures.py
```

## Test Demo
The model weights provided in `model/SIONR.pt` are the saved weights when running a random split of KoNViD-1k. The random split is shown in [data/train_val_test_split.xlsx](https://github.com/lorenzowu/SIONR/blob/master/data/train_val_test_split.xlsx), which contains video file names, scores, and train/validation/test split assignment (random).
```
python test_demo.py
```
The test results are shown in [result/test_result.xlsx](https://github.com/lorenzowu/SIONR/blob/master/result/test_result.xlsx).

## NR-VQA
|    Model   | Download            | Paper             |
|:------------:|:-------------------:|:-------------------:|
| NIQE        | [NIQE](http://live.ece.utexas.edu/research/Quality/niqe_release.zip) | [Mittal et al. IEEE SPL'12](https://ieeexplore.ieee.org/document/6353522/)
| BRISQUE        | [BRISQUE](http://live.ece.utexas.edu/research/Quality/BRISQUE_release.zip) | [Mittal et al. TIP'12](https://ieeexplore.ieee.org/document/6272356/)
| CORNIA        | [CORNIA](http://www.umiacs.umd.edu/user.php?path=pengye/research/CORNIA_release_v0.zip) | [Ye et al. CVPR'12](https://ieeexplore.ieee.org/document/6247789)
| V-BLIINDS       | [V-BLIINDS](http://live.ece.utexas.edu/research/Quality/VideoBLIINDS_Code_MicheleSaad.zip) | [Saad et al. TIP'13](https://ieeexplore.ieee.org/abstract/document/6705673/)
| HIGRADE  | [HIGRADE](http://live.ece.utexas.edu/research/Quality/VideoBLIINDS_Code_MicheleSaad.zip) | [Kundu et al. TIP'17](https://ieeexplore.ieee.org/abstract/document/7885070)
| FRIQUEE | [FRIQUEE](http://live.ece.utexas.edu/research/Quality/FRIQUEE_Release.zip) | [Ghadiyaram et al. JOV'17](https://jov.arvojournals.org/article.aspx?articleid=2599945)
| VSFA        | [VSFA](https://github.com/lidq92/VSFA) | [Li et al. ACM MM'19](https://dl.acm.org/citation.cfm?doid=3343031.3351028)
| TLVQM       | [nr-vqa-consumervideo](https://github.com/jarikorhonen/nr-vqa-consumervideo) | [Korhenen et al. TIP'19](https://ieeexplore.ieee.org/document/8742797)
