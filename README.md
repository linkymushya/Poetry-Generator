# 诗歌生成模型 Poetry Generator Module

### 声明

请注意，本项目基于Bilibili网站的<font color = "drak_blue">手写AI布丁老师</font>的代码进行了二次开发，为尊重原作者，在此对原项目的网址进行引出，并对源代码的修改处进行描述
+ **源代码网址**
  - https://github.com/shouxieai/LSTM_poetry_generate
+ **修改处**
  - 新添加了五言诗词的生成
  - 对藏头诗生成形式做了修改，允许用户随便输入任何长度的文字（源代码只能输入四个文字），以生成非四句的藏头诗
  - 对用户的交互做了一定的调整

 
### 描述

本模型致力于通过LSTM模型来对中国五言诗句和七言诗句数据集进行模型训练，从而生成五言或七言的诗歌，能够进行随机生成和藏头诗生成

### 运行环境

- 所需要的库 : torch numpy gensim

- 建议python版本：3.10

### 运行方式

- 确保`poetry_generator.py`，`poetry_5.txt`和`poetry_7.txt`在同一目录下
- 运行`poetry_generator.py`文件，会先问是生成五言还是七言，需要手动确定生成诗句类型
- 假如是生成五言诗句，如果没有文件`split_5.txt`与`word_vec_5.pkl`（七言是`split_7.txt`与`word_vec_7.pkl`），首先会运行预处理函数处理数据集，需要耐心等待
- 假如是生成五言诗句，如果没有`poetry_model_weight_5.pkl`文件（七言是`poetry_model_weight_7.pkl`文件），会先对模型进行训练，训练完会保存模型数据
- 接着会提示输入提示词，没有任何输入便是随机生成诗句，添加提示词会生成藏头诗

### Declaration
Please note that this project is based on secondary development of the code from <font color="dark_blue">手写AI布丁老师</font> of the Bilibili website. To respect the original author, the original project URL is provided below, along with descriptions of the modifications made to the source code:

+ **Source Code URL**
  - https://github.com/shouxieai/LSTM_poetry_generate

+ **Modifications**
  - Added support for generating five-character poems.
  - Modified the acrostic poem generation to allow users to input text of any length (the original code only accepted four characters), enabling the creation of non-quatrain acrostic poetry.
  - Made adjustments to user interaction.


### Description

This model is dedicated to training Chinese five character and seven character poetry datasets using LSTM models, in order to generate five character or seven character poetry, which can be randomly generated and include hidden poems

### Operating environment

- Required library: torch numpy gensim

- Suggested Python version: 3.10

### Operation mode

- Ensure that `poetry_generationor.py`, `poetry_5.txt`, and `poetry_7.txt` are in the same directory

- When running the `poerty_generator.py` file, you will first be asked whether to generate five character or seven character lines, and you need to manually determine the type of poetry to be 
generated

- If it is to generate five character poems, if there are no files `split_5.txt` and `word-vec_5.pkl` (seven character poems are `split_7.txt` and `word-vec_7.pkl`), the preprocessing function will first be run to process the dataset, and patience is required

- If it is to generate a five character poem, if there is no `poetry_madelw_eight_5.pkl` file (seven character poem is a `poetry_madelw_eight_7.pkl` file), the model will be trained first, and the model data will be saved after training

- Next, there will be a prompt to input a prompt word. Without any input, the poem will be randomly generated. Adding a prompt word will generate a hidden poem
