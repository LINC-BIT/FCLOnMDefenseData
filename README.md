# FCLOnMDefenseData

这是一个面向海防部队数据的图像分析的联邦持续学习算法系统。该系统是为应对海防作战中数据孤岛、任务动态演化与隐私安全的多重挑战，其基于“数据-算法-训练”三位一体的联邦持续学习系统架构。系统以任务粒度为核心设计维度，通过模块化分层解耦与协同控制机制， 实现海防异构数据的合规治理、动态模型的弹性进化以及跨域知识的可控共享。其主要包括数据处理功能以及联邦持续学习训练功能。数据处理模块构建军事文档解析与特征标准化流水线，将碎片化的PDF数据转化为联邦就绪的可训练图片。之后在联邦持续学习方法中支持多种中心化和去中心化的相关算法，形成“数据特征可溯源、算法能力可扩展、训练过程可管控”的闭环系统，为海防任务提供从数据感知到智能决策的全链路支撑。

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/System.png" width="100%">

# 1. PDF数据处理

## 1.1 简要介绍

数据处理模块主要是对海防场景中的各种PDF文件中的图片、表格进行提取并形成可以供联邦持续训练的图像分类数据集。其工作流程主要包括3步：

- PDF文件解析及内容提取。首先通过定位图文区域，对舰船结构图等关键图像 提取。值得注意的是，很多有效信息都存在于文档中的表格部分，为了更加有效地识 别这些信息。数据采集与预处理模块集成了 Table-Transformer，其能够针对军事表 格特性设计递归式表格提取算法：通过多阶段级联的深度学习框架，首先基于全局视觉特征定位表格区域，随后采用自注意力机制建模单元格间的层级关系，递归解析表格的物理布局(行/列/跨单元格)与逻辑语义(表头、数据、标注)。
- 图像质量检测与标签生成。会展示提取的所有图像进行检测，包括会检测一些提取的图像数据是否是有意义的，剔除无关的噪声数据。 完成质量筛查的图像将进入智能标注阶段，通过解析文件名以及表格信息能够辅助人工进行打标签。
- 图像数据增强以及标准化。会对图像进行一些翻转、模糊等图像增强的操作以便提高模型的性能。之后所有图像会设定为统一的大小以便模型进行训练。



## 1.2 使用说明

### PDF图片提取

提取基于PyPDF库进行工作，根据用户指定的输入目录文件夹（原始PDF的父文件夹）中识别图片，经过处理后会将所有图片存入到对应的输出文件夹。下面是提取核心功能，支持进行函数优化：

```python
def pdf2image1(path, pic_path):
    checkIM = r"/Subtype(?= */Image)"
    pdf = fitz.open(path)
    lenXREF = pdf.xref_length()
    count = 1
    for i in range(1, lenXREF):
        text = pdf.xref_object(i)
        isImage = re.search(checkIM, text)
        if not isImage:
						continue
        pix = fitz.Pixmap(pdf, i)
        if pix.size < 10000:  #
            continue  #
        new_name = f"img_{count}.png"
        pix.save(os.path.join(pic_path, new_name))
        # pix.writePNG()
        count += 1
        pix = None
```

或者直接运行

```shell
cd DataExtract/
python get_image.py --file_path = 'PDF文件目录' --dir_path = '图片保存目录'
```

### 图片表格识别

图片表格识别参考的是[Table-Trasnformer](https://github.com/microsoft/table-transformer/)提供的相关算法，其已经基于特定的数据集PubTables-1M进行了预训练，该数据集包括了：

- 575,305 个带注释的文档页面，包含用于表格检测的表格。
- 947,642 个带完整注释的表格，包含文本内容和完整的位置（边界框）信息，用于表格结构识别和功能分析。
- 所有表格行、列和单元格（包括空白单元格）以及其他带注释的结构（例如列标题和投影行标题）均在图像和 PDF 坐标系中提供完整的边界框。
- 所有表格和页面的渲染图像。
- 每个表格和页面图像中出现的所有单词的边界框和文本。

基于该数据集，目前提供的模型参数包括：

<b>Table Detection(表格定位检测模型参数):</b>

<table>
  <thead>
    <tr style="text-align: left;">
      <th>模型</th>
      <th>训练数据</th>
      <th>模型说明</th>
      <th>权重下载地址</th>
      <th>大小</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: left;">
      <td>DETR R18</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
  </tbody>
</table>

<b>Table Structure Recognition(表格提取参数):</b>

<table>
  <thead>
    <tr style="text-align: left;">
      <th>模型</th>
      <th>训练数据</th>
      <th>模型说明</th>
      <th>权重下载地址</th>
      <th>大小</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: left;">
      <td>TATR-v1.0</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_structure_detr_r18.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-Pub</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Pub">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Pub/resolve/main/TATR-v1.1-Pub-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-Fin</td>
      <td>FinTabNet.c</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Fin">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Fin/resolve/main/TATR-v1.1-Fin-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-All</td>
      <td>PubTables-1M + FinTabNet.c</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-All">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-All/resolve/main/TATR-v1.1-All-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
  </tbody>
</table>

用户只需要根据需求下载对应的模型参数后就能够进行训练。

之后，根据下载的模型权重可以根据表格数据进行提取并形成csv文件：

```shell
cd DataExtract/table-transformer/src/
python inference.py --image_dir = '图片文件夹' --out_dir = '输出文件目录' --model --structure_config_path = '提取配置文件目录' --structure_model_path = '提取权重目录' --detection_config_path = '检测配置文件目录' --detection_model_path = '检测模型权重'
```

**额外补充**。表格提取还支持进行微调和重新训练，如果用户想在下游任务上做的更好可以进行下面的操作。要进行训练，您需要 ```cd`` 到 ```src`` 目录并指定：1. 数据集路径，2. 任务（检测或结构），以及 3. 配置文件路径，该文件包含架构和训练所需的超参数。

训练检测模型：
```shell
python main.py --data_typedetection --config_filedetection_config.json --data_root_dir/path/to/detection_data
```

训练结构识别模型：
```shell
python main.py --data_typestructure --config_filestructure_config.json --data_root_dir/path/to/structure_data
```

如果模型训练中断，可以使用标志```--model_load_path /path/to/model.pth```并指定包含已保存优化器状态的字典文件的路径轻松恢复训练。如果您想通过微调已保存的检查点（例如```model_20.pth```）来重新开始训练，请使用标志```--model_load_path /path/to/model_20.pth```和标志```--load_weights_only```来指示恢复训练不需要之前的优化器状态。无论是微调还是从头开始训练新模型，您都可以选择创建一个新的配置文件，其中包含与我们使用的默认训练参数不同的参数。使用以下命令指定新的配置文件：```--config_file /path/to/new_structure_config.json```。创建新的配置文件很有用，例如，如果您想在微调期间使用不同的学习率```lr```。

### 图像增强和标准化

提取到的图片可以运行下面的代码进行处理，运行后会对图片继续旋转、模糊、变色等操作，之后形成统一的

```shell
cd DataExtract/
python ImageEnhance.py --data_dir = '原始图片存放路径' --output_dir = '增强图像存放路径'
```



# 2. 支持的相关联邦学习算法

## 2.1 支持的中心化联邦学习算法

目前系统支持了很多经典的和最新的联邦学习算法，包括基于异构网络模型的联邦学习算法以及聚类联邦学习算法，主要有：

- **[FedMD](https://arxiv.org/abs/2107.08517)**：这篇论文来自 AIR(2017)。它使用公共数据集在聚合过程中更新蒸馏模型。方法描述可在[此处](Baselines/FedMD)找到
- **[FedKD](https://arxiv.org/abs/2003.13461)**：这篇论文来自 AIR(2017)。它根据客户端的网络层设计了各种蒸馏损失。方法描述可在[此处](Baselines/FedKD)找到。
- **[FedKEMF](https://proceedings.mlr.press/v139/collins21a.html)**：这篇论文来自 ICML(2021)。它考虑在聚合过程中合并所有教师网络，并使用通用数据集来蒸馏出更好的服务器端全局网络。您可以在[此处](Baselines/FedKD)找到方法描述。
- **[FedGKT](https://proceedings.neurips.cc/paper/2020/hash/a1d4c20b182ad7137ab3606f0e3fc8a4-Abstract.html)** ：本文来自 NIPS (2023)。它设计了一种交替最小化方法的变体，用于在边缘节点上训练小型模型，并通过知识蒸馏定期将其知识迁移到大型服务器端模型。您可以在[此处](Baselines/FedGKT)找到方法描述。
- **[CFL](https://ieeexplore.ieee.org/abstract/document/9174890)** ：本文来自 NNLS (Volume: 32, 2020)。它根据网格参数的余弦相似度将客户端划分到不同的簇中，并对同一簇中的客户端进行全局聚合。方法描述可在[此处](Baselines/FedKD)找到。
- **[IFCA](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html)**：本文来自 NIPS (2020)。它估计客户端的聚类特征，优化每个簇的模型参数，并允许不同簇之间共享参数。您可以在[此处](Baselines/IFCA)找到方法描述
- **[GradMFL](https://link.springer.com/chapter/10.1007/978-3-030-95384-3_38)** ：本文来自 ICAAPP (2021)。它引入了一个层次聚类来组织客户端，并支持不同层次结构之间的知识迁移。您可以在[此处](Baselines/GradMFL)找到方法描述。

## 2.2 支持的去中心化联邦学习算法

目前系统还支持了多个去中心化的联邦学习算法，主要有：

- **[PENS](https://arxiv.org/abs/2107.08517)**：该论文来自 ICML(2018)。它提出了基于性能的邻居选择 (PENS) 方法，其中去中心化联邦学习系统中的客户端会评估彼此数据的训练损失，以识别具有相似数据分布的对等节点。这种有针对性的对等节点选择使客户端能够以完全去中心化的方式协作更新其模型。您可以在[此处](baselines/PENS) 找到方法描述。
- **[HDFL](https://ieeexplore.ieee.org/abstract/document/10226164)**：该论文来自 INFOCOM(2023)。它引入了一个集成的分层去中心化联邦学习 (HDFL) 框架，其中不同单元中的设备会定期达成单元内 D2D 共识，然后进行单元间聚合，以协作训练全局模型。这种分层方法旨在优化收敛速度，同时平衡多单元场景下的通信和能耗开销。您可以在[此处](baselines/HDFL) 找到方法描述。
- **[FedPC](https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Yuan_Peer-to-Peer_Federated_Continual_Learning_for_Naturalistic_Driving_Action_Recognition_CVPRW_2023_paper.html)**：这篇论文来自 CVPR(2023)。它提出了一种新颖的点对点联邦持续学习框架，使客户端能够使用流式驾驶数据持续更新其本地模型，并直接与对等节点交换模型更新，从而无需中央服务器。您可以在[此处](baselines/FedPC)找到方法描述。
- **[DPFL](https://ieeexplore.ieee.org/abstract/document/9993756/)**：本文来自TMC（第23卷，2024年）。它提出了一种用于6G无线网络的集成式分层去中心化联邦学习框架。其中每个小区中的设备会定期达成D2D共识，然后参与小区间聚合，从而联合训练全局模型，同时优化收敛速度和通信能耗权衡。您可以在[此处](baselines/DPFL)找到方法描述。
- **[FedIR](https://ieeexplore.ieee.org/abstract/document/9944948/)**：本文来自TMC（第23卷，2024年）。它提出了一种新颖的方法，利用两阶段优化方法来平衡联邦学习中的系统延迟和能耗，其中本地模型通过分布式共识机制进行协作更新，无需中央服务器。您可以在[此处](baselines/FedIR) 找到该方法的描述。

## 2.3 支持的联邦持续学习算法

- **[FedKNOW](https://ieeexplore.ieee.org/abstract/document/10184531/)**：本文摘自 ICDE (2023)。它提出了一种新颖的通信高效的联邦学习算法，该算法采用自适应梯度量化和选择性客户端聚合，根据网络状况和客户端异构性动态调整模型更新，从而降低通信开销并加速收敛。您可以在[此处](baselines/FedKNOW) 找到方法描述。
- **[FedViT](https://www.sciencedirect.com/science/article/abs/pii/S0167739X23004879)**：本文摘自《Future Generation Computer Systems 》（第154卷，2024年）。它提出了一种新颖的集成优化框架，该框架将先进的机器学习与启发式搜索方法相结合，通过自适应迭代参数调整来动态优化复杂的工业系统。您可以在[此处](baselines/FedViT)找到方法描述
- **[FedCL](https://ieeexplore.ieee.org/abstract/document/9190968/)**：本文来自 ICIP (2020)。它提出了一种新颖的联邦学习框架，该框架集成了区块链技术，以确保客户端之间模型更新的安全性和去中心化，从而增强数据隐私和系统稳健性。您可以在[此处](baselines/FedCL)找到方法描述
- **[FedWEIT](https://proceedings.mlr.press/v139/yoon21b.html?ref=https://githubhelp.com)**：本文来自 ICML (2021)。它提出了一种新颖的方法，利用自监督学习，通过在元训练阶段有效利用未标记数据来提升少样本学习模型的性能。您可以在[此处](baselines/WEIT)找到方法描述。
- **[Cross-FCL](https://ieeexplore.ieee.org/abstract/document/9960821/)**：本文来自TMC（第23卷，2024年）。它提出了一种新颖的联邦学习框架，该框架集成了区块链技术，以确保客户端之间安全且去中心化的模型更新，从而增强数据隐私和系统稳健性。您可以在[此处](baselines/Cross_FCL)找到方法描述。
- **[TFCL](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Traceable_Federated_Continual_Learning_CVPR_2024_paper.html)**：本文来自CVPR（2024年）。它提出了一种新颖的可追踪联邦持续学习 (TFCL) 范式，引入了 TagFed 框架，该框架将模型分解为针对每个客户端任务的标记子模型，从而实现精确追踪和选择性联邦，从而有效地处理重复性任务。您可以在 [此处](baselines/TFCL) 找到方法描述。
- **[Loci](https://ieeexplore.ieee.org/abstract/document/10857343/)**：本文来自 TPDS（第 36 卷，2025 年）。它提出 Loci 使用紧凑的模型权重来抽象客户端过去和同伴的任务知识，并开发一种通信高效的方法来训练每个客户端的本地模型，方法是将其任务知识与其他客户端中最准确的相关知识进行交换。您可以在 [此处](baselines/Loci) 找到方法描述。

# 3. 方法运行例子

## 3.1 如何开始

**Requirements**

- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+

**准备虚拟环境**

1. 创建一个conda环境并激活它。

```shell
conda create -n Loci python=3.7
conda active FCLOnMDefenseData
```

2. 下载PyTorch 1.9+ 在[官方网址](https://pytorch.org/). 建议使用带有CUDA的Pytorch版本（本机要带有NVIDIA或者AMD的显卡）

![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

3. 克隆本项目并下载对应的依赖：

```shell
git clone https://github.com/LINC-BIT/FCLOnMDefenseData.git
pip install -r requirements.txt
```

## 3.2 各方法运行例子

**PENS**

```shell
python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=resnet --gpu=0 --seed=614 --m_ft=300
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/PENS.png" width="100%">

**FedIR**

```shell
python TMC/FedIR/mainFedIR.py --alg=FedIR --model=mobinet --gpu=0 --seed=614 --m_ft=300
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedIR.png" width="100%">

**DPMN**

```shell
python TMC/DPMN/mainDPMN.py --alg=DPMN --model=resnet --gpu=0 --seed=614 --m_ft=300
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/DPMN.png" width="100%">

**FedKNOW**

```shell
python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=resnet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedKNOW.png" width="100%">

**FedViT**

```shell
python FCL/FedViT/mainFedViT.py --alg=FedViT --model=tiny_pit --seed=614 --epochs=120 --task=20 --m_ft=500 --gpu=0 --n_memories=40 --round=5 --local_ep=5 --shard_per_user=8
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedViT.png" width="100%">

**PuzzleFL**

```shell
python ClientTrainOur/mainOur_EWC.py --alg=EWC_Our --model=tiny_vit --gpu=0 --seed=614
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/PuzzleFL.png" width="100%">

**FedWEIT**

```shell
python FCL/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_cnn --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedWEIT.png" width="100%">

**FedMD**

```
python FL/FedMD/mainFedMD.py --alg=EWC_Our --model=resnet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedMD.png" width="100%">

**FedKD**

```shell
python FL/FedKD/mainFedKD.py --alg=EWC_Our --model=resnet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedKD.png" width="100%">

**FedKEMF**

```shell
python FL/FedKEMF/mainFedKEMF.py --alg=EWC_Our --model=resnet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/FedKEMF.png" width="100%">

**Loci**

```shell
python FCL/FedKEMF/Loci.py --alg=EWC_Our --model=resnet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100
```

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/running_image/Loci.png" width="100%">