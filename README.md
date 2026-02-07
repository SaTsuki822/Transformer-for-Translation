## Introduction
结合开源项目自己敲了一个transformer项目，用于中英文本翻译任务。代码里添加了详细的注释。

目前训练脚本只用loss值作为评价指标。

## How to use
### 环境配置

创建环境、安装pytorch
```
conda create -n transformer python==3.10

conda activate transformer

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 # CUDA 12.1

pip install -r requirements.txt
```

环境参考：
```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
_openmp_mutex             5.1                       1_gnu    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
bzip2                     1.0.8                h5eee18b_6    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ca-certificates           2025.12.2            h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
certifi                   2022.12.7                pypi_0    pypi
charset-normalizer        2.1.1                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
expat                     2.7.4                h7354ed3_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
filelock                  3.20.0                   pypi_0    pypi
fsspec                    2025.12.0                pypi_0    pypi
idna                      3.4                      pypi_0    pypi
jieba                     0.42.1                   pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
ld_impl_linux-64          2.44                 h153f514_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libexpat                  2.7.4                h7354ed3_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libffi                    3.4.4                h6a678d5_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc                    15.2.0               h69a1729_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc-ng                 15.2.0               h166f726_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgomp                   15.2.0               h4751f2c_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libnsl                    2.0.0                h5eee18b_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx                 15.2.0               h39759b7_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx-ng              15.2.0               hc03a8fd_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuuid                   1.41.5               h5eee18b_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libxcb                    1.17.0               h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libzlib                   1.3.1                hb25bd0a_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lxml                      6.0.2                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.5                  h7934f7d_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
networkx                  3.4.2                    pypi_0    pypi
numpy                     1.26.0                   pypi_0    pypi
openssl                   3.0.19               h1b28b03_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
packaging                 25.0            py310h06a4308_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pandas                    2.3.3                    pypi_0    pypi
pillow                    12.0.0                   pypi_0    pypi
pip                       25.3               pyhc872135_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
portalocker               3.2.0                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python                    3.10.19              h6fa692b_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
readline                  8.3                  hc2a1206_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
regex                     2026.1.15                pypi_0    pypi
requests                  2.28.1                   pypi_0    pypi
setuptools                80.10.2         py310h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
six                       1.17.0                   pypi_0    pypi
sqlite                    3.51.1               he0a8d7e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
sympy                     1.14.0                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
tk                        8.6.15               h54e0aa7_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
torch                     2.1.0+cu118              pypi_0    pypi
torchaudio                2.1.0+cu118              pypi_0    pypi
torchvision               0.16.0+cu118             pypi_0    pypi
tqdm                      4.67.3                   pypi_0    pypi
triton                    2.1.0                    pypi_0    pypi
typing-extensions         4.15.0                   pypi_0    pypi
tzdata                    2025.3                   pypi_0    pypi
urllib3                   1.26.13                  pypi_0    pypi
wheel                     0.46.3          py310h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libx11               1.8.12               h9b100fa_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libxau               1.0.12               h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libxdmcp             1.1.5                h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-xorgproto            2024.1               h5eee18b_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xz                        5.6.4                h5eee18b_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zlib                      1.3.1                hb25bd0a_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```
### 数据集准备
数据集来源：[WMT 2024 Translation Task Training Data](https://www2.statmt.org/wmt24/mtdata/)

用到了WMT里面的一个新闻评论中英对照数据集，相应链接为：[Index of /news-commentary/v18.1/training](https://data.statmt.org/news-commentary/v18.1/training/)

源文件为tsv格式，包含450k对中英文本对。

下载后，使用本项目里的dataProcess.py脚本分割为train.tsv valid.tsv和test.tsv三个文件。

如果觉得这么大的数据集训练起来太耗时间，可以用dataSplit.py脚本再处理成较小的文件。
### 创建词汇表
```
python util/buildVocab.py
```
在buildVocab.py里指定数据集源文件位置，创建整个数据集的词汇表，防止在后续训练和推理中出现<unk>问题

后续的训练和推理过程中不再创建新的词汇表，直接调用现在创建好的这个词汇表

### 修改配置
首先在options/config.yaml里修改自己的数据集路径、模型参数、训练超参数和日志打印/模型验证频率等配置。

模型推理前在options/inference_config.yaml中修改预训练权重的路径
### 训练
```
python train.py
```
### 推理

```
python inference.py
```