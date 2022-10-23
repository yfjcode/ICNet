**ICNet****模型 自验报告**

院校：四川大学



**1.** **模型简介**

**① 模型结构（说明：简单描述，方便验收人员了解模型基本概况）**

ICNet模型于2018年在论文《ICNet for Real-Time Semantic Segmentation on High-Resolution Images》中被提出，该模型提出是为了解决现有的方法对于像素级分割很难在较大比例上减少运算的计算量的问题，主要用于实时的语义分割任务。ICNet采用多分辨率的分支构建语义分割模型，它将图像分为高中低三层，性能大幅提升的根本原因是让低分辨的图像经过语义分割网络产生粗糙的分割结果；之后特征级联混合单元（cascade label guidance）与标签引导的级联策略（cascade label guidance strategy）将中分辨率和高分辨率的特征整合，逐步地优化之前生成的粗糙分割结果。

 

**② 数据集（说明：如需填写多个数据集地址，可在代码的 read.md 中填写获取和下载数据集的方式）**

Cityscapes数据集，以下是在ModelArts平台上数据集的目录，由于我们是在原本数据集有1个多G，全部拿来跑得话，很容易跑满内存，故我们就选择一个城市的一些图片完成。都只选了一张图片，但是我们实际测试过，放全部图片跑代码也是不会有任何问题和报错的：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)

​    -训练集：1张图片

​    -测试集： 1张图片

-验证集：1张图片

 

**2.** **自验结果**

**（根据实测结果进行截图，MindSpore版本请使用当期启动项目时候最新的版本，且精度无明显错误）**

**版本：**MindSpore 1.8.0

**GPU** **环境：** Nvidia CUDA10.1+GeForce RTX 1080

训练结果：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image003.jpg)

 

验证结果：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image005.jpg)

**3.** **Notebook****执行自验截图**

**（说明：无报错，且端到端可执行）**

 **·核心公式&讲解**

**1.**  **CFF(cascade feature fusion)****模块**

不同的cascade feature使用论文提出的 Cascade Feature Fusion Unit（CFF）进行融合。F1其中上采样使用双线性插值法。然后使用空洞卷积继续上采样，相比于deconvolution，这种方法只需要少量的卷积核就能获得相同的感受野。而F2则进行了一个projection conv的操作。 

其中 F1和F2是两个输入特征图，尺寸分别为（C1，H1，W1）和（C2，H2，W2）；LABEL是ground-truth的标签，尺寸是（1，H2，W2）。

![IMG_256](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image007.png)

 

**2.**  **Cascade Label Guidance****级联的标签指导**

为了提高F1、F2的学习能力，作者在F1、F2这里使用了cascade label guidance。使用不同尺寸的Groud-Truth去引导低、中、高分辨率学习。对结构中的每一个分支（不同大小的Input）用带有权重的softmax交叉熵损失函数算Loss，从而对每一个分支都能进行很好的训练。公式如下：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg)

**·模型图**

![1663575886933](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image011.png)

**·训练评估逻辑讲解**

环境准备与数据读取

本案例基于MindSpore-GPU版本实现，在GPU上完成模型训练。

案例实现所使用的数据:Cityscape Dataset Website。

为了下载数据集，我们首先需要在Cityscapes数据集官网进行注册，并且最好使用edu教育邮箱进行注册，此后就可以下载数据集了，这里我们下载了两个文件：gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip (11GB)。

下载完成后，我们对数据集压缩文件进行解压，文件的目录结构如下所示。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)

由于原本数据集有1个多G，全部拿来跑得话，很容易掉卡，故我们就选择一个城市的一些图片完成。

首先要处理数据，生成对应的.mindrecord和.mindrecord.db文件。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image013.png)

需要注意的是，在生成这两个文件之前，我们要建立一个文件夹，用cityscapes_mindrecord命名，放在cityscapes文件夹的同级目录下。而且要保持cityscapes_mindrecord文件夹里面为空。

 

此外，由于ICNet的案例实现是基于ResNet50的骨干网络进行实现的，所以在训练ICNet之前往往需要先跑一边ResNet50的代码，故本项目会先将ResNet50的模型跑好，并将其保存在对应路径之中，在进行ICNet代码编写过程中直接在src资源包中调用即可，就不用在重新训练一遍ResNet50模型了。预训练模型位置在/home/ma-user/work/ICNet/root/cacheckpt/里面，如下图所示：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png)

模型训练部分：

一些参数：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg)

ICNet主要结构：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image017.png)

PPM结构：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png)

Head结构：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image020.jpg)

CFF结构：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image022.jpg)

 

自测训练结果：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image023.jpg)

 

模型验证部分：

评价指标为:mIOU、pixAcc、average_time.

自测验证结果：

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image024.jpg)

具体代码地址在ModelArts平台已经部署，网址为：

https://authoring-dev.cncentral231.xckpjs.com/87136e07-4a5f-4514-8285-82551fab3b15/lab/tree/ICNet/train.ipynb

gitee网址也已经提交，网址为：

https://gitee.com/yangfj/icnet_v1