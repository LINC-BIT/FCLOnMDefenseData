
面向海防部队数据的图像分析的联邦持续学习算法系统。该系统是为应对海防作战中数据孤岛、任务动态演化与隐私安全的多重挑战，其基于“数据-算法-训练”三位一体的联邦持续学习系统架构。系统以任务粒度为核心设计维度，通过模块化分层解耦与协同控制机制， 实现海防异构数据的合规治理、动态模型的弹性进化以及跨域知识的可控共享。其主要包括数据处理功能以及联邦持续学习训练功能。数据处理模块构建军事文档解析与特征标准化流水线，将碎片化的PDF数据转化为联邦就绪的可训练图片。之后在联邦持续学习方法中支持多种中心化和去中心化的相关算法，形成“数据特征可溯源、算法能力可扩展、训练过程可管控”的闭环系统，为海防任务提供从数据感知到智能决策的全链路支撑。

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



## 1.3 目前收集的海防数据集展示

目前收集了人物、装备、物质、设备、环境8个任务下的总共47个类别的数据，相关数据已经存放到[网盘](https://pan.baidu.com/s/1iSnVEfUtihx3MMfWfhdR6A ),密码为 ba7d。主要有以下的几个类别：

1. 人员/训练图片类别（任务1）
   - 单兵战术训练（掩体射击、匍匐前进）
   - 小组战术协同（CQB室内近战、野外伏击）
   - 水下作战训练（潜水装备穿戴、水下拆弹）
   - 直升机索降训练（绳索速降、甲板着陆）
   - 军犬协同巡逻（海岸嗅探、船只搜查）
   - 医疗救援演练（伤员搬运、海上急救）
   - 军民联合防灾（台风救援、渔民疏散）
2. 舰船与巡逻工具（任务2）
   - 近海巡逻艇（22型导弹艇、双体快艇）
   - 大型巡逻舰（海警船、海防驱逐舰）
   - 隐身高速拦截艇（低雷达反射外形）
   - 气垫登陆艇（沙滩突击、快速登岸）
   - 无人侦察艇（太阳能驱动、摄像头搭载）
   - 反潜巡逻舰（拖曳声呐、深弹发射器）
   - 特种潜水器（蛙人输送艇、微型潜艇）
3. 武器装备（任务3）
   - 岸基反舰导弹（鹰击-62发射车、固定发射井）
   - 单兵火箭筒（PF-98反坦克火箭筒）
   - 舰载近防炮（H/PJ-11型11管30mm舰炮）
   - 水雷布设装置（锚雷、沉底雷投放）
   - 便携式防空导弹（FN-6单兵防空系统）
   - 电子干扰设备（舰载雷达干扰机）
   - 水下声呐阵列（海底固定监听设备）
4. 基地设施（任务4）
   - 海岸雷达站（旋转雷达罩、相控阵雷达）
   - 地下指挥所（防爆门、作战指挥屏幕）
   - 码头防御工事（防波堤机枪位、沙袋掩体）
   - 移动式哨所（集装箱改装哨塔、车载哨站）
   - 弹药储存库（防爆墙、温湿度控制系统）
   - 直升机停机坪（舰载/陆基甲板标识）
5. 作战地理环境（任务5）
   - 岩礁海岸（悬崖哨所、礁石巡逻路径）
   - 滩涂湿地（红树林隐蔽据点、淤泥行军）
   - 人工岛礁（填海造岛设施、灯塔）
   - 近海养殖区（渔排监视、防破坏巡逻）
   - 海峡要道（航道管制、船舶引导）
6. 可疑目标（任务6）
   - 改装间谍渔船（伪装网、隐藏天线）
   - 高速走私快艇（大马力外挂发动机）
   - 低空突防无人机（四旋翼侦查机）
   - 仿生机器人（鱼形水下探测器）
   - 漂流水雷（球形触发式水雷）
   - 偷渡橡皮艇（超载人员、简易动力）
7. 服装装备（任务7）
   - 作战服（海洋迷彩、荒漠迷彩）
   - 救生装具（充气式救生衣、信号发射器）
   - 潜水装备（密闭循环呼吸器、脚蹼）
   - 防弹护具（四级防弹插板、战术头盔）
   - 夜战装备（单目夜视仪、IR标识贴）
8. 特殊天气（任务8）
   - 大雾警戒（雷达扫描界面特写）
   - 极寒海冰（破冰船作业、冰面巡逻）
   - 暴雨巨浪（船舱进水抢险）
   - 节日战备（春节/国庆执勤场景）
<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/海防数据集.jpg" width="100%">

# 2. 支持的模型

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/Model.png" width="100%">

目前系统支持了包括图像分类、文本分类和表格类数据预测的相关模型，具体如下：

## 2.1 图像分类模型

对于图像分类模型来说，主要支持了包括传统的CNN系列、ResNet系列以及ViT系列的模型。

- [CNN](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf): CNN（卷积神经网络）是一种专用于处理网格数据（如图像）的深度学习模型，通过局部感知和权值共享高效提取特征。其核心结构包含卷积层、池化层和全连接层。目前系统设置了六层CNN以及十层CNN模型。

- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)：该模型由多个卷积层和池化层组成，用于提取图像中的信息。通常，当网络深度较大时，ResNet 容易出现梯度消失（爆炸）现象，性能下降。因此，ResNet 添加了 BatchNorm 来缓解梯度消失（爆炸），并添加了残差连接来缓解性能下降。
- [MobileNet](https://arxiv.org/abs/1801.04381)：MobileNet 是一个轻量级卷积网络，广泛使用了深度可分离卷积。
- [DenseNet](https://arxiv.org/pdf/1707.06990.pdf)：DenseNet 扩展了 ResNet，通过在各个块之间添加连接来聚合所有多尺度特征。
- [WideResNet](https://arxiv.org/pdf/1605.07146)：WideResNet（宽残差网络）是一种深度学习模型，它基于 ResNet 架构，通过增加残差块的宽度（使用更多特征通道）来提升性能和效率，同时降低网络深度。
- [Vit](https://arxiv.org/abs/2010.11929)：Vision Transformer（ViT）将 Transformer 架构应用于图像识别任务。它将图像分割成多个块，然后将这些小块作为序列数据输入到 Transformer 模型中，利用自注意力机制捕捉图像中的全局和局部信息，从而实现高效的图像分类。目前系统支持了TinyPiT以及TinyViT两个网路。

## 2.2 文本分类模型

对于图像分类模型来说，主要支持了包括传统的RNN系列以及Transformer系列模型。

- [RNN](https://arxiv.org/pdf/1406.1078)：RNN（循环神经网络）是一种专为序列数据设计的神经网络，擅长处理时间序列和具有时间依赖性的自然语言。

- [LSTM](https://arxiv.org/pdf/1406.1078)：LSTM（长短期记忆网络）是一种特殊的 RNN，可以学习长期依赖关系，适用于时间序列分析和语言建模等任务。
- [Bert](https://arxiv.org/abs/1810.04805)：BERT（基于 Transformer 的双向编码器表示）是一种基于 Transformer 架构的预训练语言表示模型，它通过深度双向训练来捕获文本中的上下文信息。BERT 模型在自然语言处理 (NLP) 任务中表现出色，可用于文本分类、问答系统和命名实体识别等各种应用。
- [LSTMMoE](https://readpaper.com/paper/2952339051):LSTMMoE（LSTM with Mixture of Experts）是一种将长短期记忆 (LSTM) 网络与混合专家框架相结合的模型，通过针对不同输入模式动态选择专门的专家网络来增强序列建模。
- [GPT2](https://github.com/openai/gpt-2): GPT-2（生成式预训练 Transformer 2）是由 OpenAI 开发的大规模语言模型，旨在通过利用基于 Transformer 的架构并在各种互联网数据上进行预训练来生成连贯且与上下文相关的文本。
- [GPTNeo](https://github.com/EleutherAI/gpt-neox): GPT-Neo 是由 EleutherAI 开发的开源语言模型，旨在作为 GPT 的替代方案，利用基于 Transformer 的架构生成高质量、与上下文相关的文本。

## 2.3 表格预测模型

对于表格预测来说，主要支持了神经网络系列以及树模型系列的网络。

- [全连接神经网络](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)：DNN（全连接神经网络）是一种基础的深度学习模型，其内部逻辑是通过多层神经元之间的全连接权重矩阵进行非线性变换，每一层的输出作为下一层的输入，最终实现输入到输出的复杂映射。它依靠反向传播算法优化权重，利用激活函数（如ReLU、Sigmoid）引入非线性，适用于各类通用机器学习任务。
- [XGBoost](https://arxiv.org/abs/1603.02754)：XGBoost（极端梯度提升）是一种基于决策树的集成学习算法，其内部逻辑通过迭代地训练弱分类器（CART树），并利用梯度提升框架（GBDT）优化损失函数，同时引入正则化项控制模型复杂度以防止过拟合。
- [LightGBM](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)：LightGBM是一种高效的梯度提升框架，其核心逻辑基于直方图算法（将连续特征离散化为 bins 加速计算）和单边梯度采样（GOSS），保留大梯度样本并随机采样小梯度样本以提升训练速度。同时，它采用Leaf-wise 生长策略（仅分裂增益最大的叶子节点，而非 Level-wise），在降低计算开销的同时保持较高精度，尤其适合大规模数据和高维特征场景。

# 3. 支持的相关联邦学习算法

## 3.1 支持的中心化联邦学习算法

目前系统支持了很多经典的和最新的联邦学习算法，包括基于异构网络模型的联邦学习算法以及聚类联邦学习算法，主要有：

- **[FedMD](https://arxiv.org/abs/2107.08517)**：这篇论文来自 AIR(2017)。它使用公共数据集在聚合过程中更新蒸馏模型。方法描述可在[此处](Baselines/FedMD)找到
- **[FedKD](https://arxiv.org/abs/2003.13461)**：这篇论文来自 AIR(2017)。它根据客户端的网络层设计了各种蒸馏损失。方法描述可在[此处](Baselines/FedKD)找到。
- **[FedKEMF](https://proceedings.mlr.press/v139/collins21a.html)**：这篇论文来自 ICML(2021)。它考虑在聚合过程中合并所有教师网络，并使用通用数据集来蒸馏出更好的服务器端全局网络。您可以在[此处](Baselines/FedKD)找到方法描述。
- **[FedGKT](https://proceedings.neurips.cc/paper/2020/hash/a1d4c20b182ad7137ab3606f0e3fc8a4-Abstract.html)** ：本文来自 NIPS (2023)。它设计了一种交替最小化方法的变体，用于在边缘节点上训练小型模型，并通过知识蒸馏定期将其知识迁移到大型服务器端模型。您可以在[此处](Baselines/FedGKT)找到方法描述。
- **[CFL](https://ieeexplore.ieee.org/abstract/document/9174890)** ：本文来自 NNLS (Volume: 32, 2020)。它根据网格参数的余弦相似度将客户端划分到不同的簇中，并对同一簇中的客户端进行全局聚合。方法描述可在[此处](Baselines/FedKD)找到。
- **[IFCA](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html)**：本文来自 NIPS (2020)。它估计客户端的聚类特征，优化每个簇的模型参数，并允许不同簇之间共享参数。您可以在[此处](Baselines/IFCA)找到方法描述
- **[GradMFL](https://link.springer.com/chapter/10.1007/978-3-030-95384-3_38)** ：本文来自 ICAAPP (2021)。它引入了一个层次聚类来组织客户端，并支持不同层次结构之间的知识迁移。您可以在[此处](Baselines/GradMFL)找到方法描述。

## 3.2 支持的去中心化联邦学习算法

目前系统还支持了多个去中心化的联邦学习算法，主要有：

- **[PENS](https://arxiv.org/abs/2107.08517)**：该论文来自 ICML(2018)。它提出了基于性能的邻居选择 (PENS) 方法，其中去中心化联邦学习系统中的客户端会评估彼此数据的训练损失，以识别具有相似数据分布的对等节点。这种有针对性的对等节点选择使客户端能够以完全去中心化的方式协作更新其模型。您可以在[此处](baselines/PENS) 找到方法描述。
- **[HDFL](https://ieeexplore.ieee.org/abstract/document/10226164)**：该论文来自 INFOCOM(2023)。它引入了一个集成的分层去中心化联邦学习 (HDFL) 框架，其中不同单元中的设备会定期达成单元内 D2D 共识，然后进行单元间聚合，以协作训练全局模型。这种分层方法旨在优化收敛速度，同时平衡多单元场景下的通信和能耗开销。您可以在[此处](baselines/HDFL) 找到方法描述。
- **[FedPC](https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Yuan_Peer-to-Peer_Federated_Continual_Learning_for_Naturalistic_Driving_Action_Recognition_CVPRW_2023_paper.html)**：这篇论文来自 CVPR(2023)。它提出了一种新颖的点对点联邦持续学习框架，使客户端能够使用流式驾驶数据持续更新其本地模型，并直接与对等节点交换模型更新，从而无需中央服务器。您可以在[此处](baselines/FedPC)找到方法描述。
- **[DPFL](https://ieeexplore.ieee.org/abstract/document/9993756/)**：本文来自TMC（第23卷，2024年）。它提出了一种用于6G无线网络的集成式分层去中心化联邦学习框架。其中每个小区中的设备会定期达成D2D共识，然后参与小区间聚合，从而联合训练全局模型，同时优化收敛速度和通信能耗权衡。您可以在[此处](baselines/DPFL)找到方法描述。
- **[FedIR](https://ieeexplore.ieee.org/abstract/document/9944948/)**：本文来自TMC（第23卷，2024年）。它提出了一种新颖的方法，利用两阶段优化方法来平衡联邦学习中的系统延迟和能耗，其中本地模型通过分布式共识机制进行协作更新，无需中央服务器。您可以在[此处](baselines/FedIR) 找到该方法的描述。

## 3.3 支持的联邦持续学习算法

- **[FedKNOW](https://ieeexplore.ieee.org/abstract/document/10184531/)**：本文摘自 ICDE (2023)。它提出了一种新颖的通信高效的联邦学习算法，该算法采用自适应梯度量化和选择性客户端聚合，根据网络状况和客户端异构性动态调整模型更新，从而降低通信开销并加速收敛。您可以在[此处](baselines/FedKNOW) 找到方法描述。
- **[FedViT](https://www.sciencedirect.com/science/article/abs/pii/S0167739X23004879)**：本文摘自《Future Generation Computer Systems 》（第154卷，2024年）。它提出了一种新颖的集成优化框架，该框架将先进的机器学习与启发式搜索方法相结合，通过自适应迭代参数调整来动态优化复杂的工业系统。您可以在[此处](baselines/FedViT)找到方法描述
- **[FedCL](https://ieeexplore.ieee.org/abstract/document/9190968/)**：本文来自 ICIP (2020)。它提出了一种新颖的联邦学习框架，该框架集成了区块链技术，以确保客户端之间模型更新的安全性和去中心化，从而增强数据隐私和系统稳健性。您可以在[此处](baselines/FedCL)找到方法描述
- **[FedWEIT](https://proceedings.mlr.press/v139/yoon21b.html?ref=https://githubhelp.com)**：本文来自 ICML (2021)。它提出了一种新颖的方法，利用自监督学习，通过在元训练阶段有效利用未标记数据来提升少样本学习模型的性能。您可以在[此处](baselines/WEIT)找到方法描述。
- **[Cross-FCL](https://ieeexplore.ieee.org/abstract/document/9960821/)**：本文来自TMC（第23卷，2024年）。它提出了一种新颖的联邦学习框架，该框架集成了区块链技术，以确保客户端之间安全且去中心化的模型更新，从而增强数据隐私和系统稳健性。您可以在[此处](baselines/Cross_FCL)找到方法描述。
- **[TFCL](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Traceable_Federated_Continual_Learning_CVPR_2024_paper.html)**：本文来自CVPR（2024年）。它提出了一种新颖的可追踪联邦持续学习 (TFCL) 范式，引入了 TagFed 框架，该框架将模型分解为针对每个客户端任务的标记子模型，从而实现精确追踪和选择性联邦，从而有效地处理重复性任务。您可以在 [此处](baselines/TFCL) 找到方法描述。
- **[Loci](https://ieeexplore.ieee.org/abstract/document/10857343/)**：本文来自 TPDS（第 36 卷，2025 年）。它提出 Loci 使用紧凑的模型权重来抽象客户端过去和同伴的任务知识，并开发一种通信高效的方法来训练每个客户端的本地模型，方法是将其任务知识与其他客户端中最准确的相关知识进行交换。您可以在 [此处](baselines/Loci) 找到方法描述。

# 4. 方法运行例子

## 4.1 如何开始

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

## 4.2 各方法运行例子

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

<img width="274" height="94" alt="image" src="https://github.com/user-attachments/assets/988e7931-a08d-4e21-a910-e07705727dfc" /># FCLOnMDefenseData

