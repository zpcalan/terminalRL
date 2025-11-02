# terminalRL

### 目标确定
- 训练：基于AReal构建一个基于终端指令的AI agent。（代码已实现）✔️
- 评测：基于terminal-bench测试集进行评测。（时间原因还未评测）❌
----

### 数据集选择

训练采用github上开源项目https://github.com/Danau5tin/terminal-bench-rl，数据在https://github.com/Danau5tin/terminal-bench-rl/blob/main/dataset/latest_verified.csv。

----

### 训练实现
流程图可参考`doc/flowchart.excalidraw`
1. 参考了ASearcher代码仓以及AReal中已有的MultiTurn训练实现流程，实现了TerminalAgent以及TerminalMultiturnRLVRWorkflow。
2. 整体训练流程：
    > 1. 根据`task_id`字段构建独立每个任务的镜像，一次构建后重复使用进行训练。
    > 2. 解析`prompt`字段，输入RolloutEngine，并且根据输出结果解析下一步shell指令并使用docker python SDK`docker.container.run`执行指令，收集console输出结果。
    > 3. 执行python用例，即`test_functions`字段校验结果，并根据`test_weights`分配奖励。
    > 4. 重复step2和step3，完成训练。
----

### 快速执行
```
#根据个人配置修改`terminal_rl.yaml`中的`train_dataset.path`, `valid_dataset.path`以及`actor.path`字段
cd train
bash train.sh 
```

### 实验结果
实验结果截图：
![image](./output.jpeg)

- 注意：因为时间原因，reward方法还未充分和测试验证，大部分场景下，reward输出为0。

### 实验分析以及TODO List
1. 设计更加平滑的奖励函数，不止通过最终`test_functions`结果，还需要根据各轨迹的console output进行判断。
2. 针对特定样本，其dockerfile可能在我目前的环境中执行存在问题，需要搭建一个更通用的docker环境以支持样本中的镜像能成功创建。
3. test_function在镜像中执行可能存在问题，部分镜像没有安装python以及pytest，需要优化训练数据，使得能够在宿主机进行测试验证。
4。设计更完善的system prompt，目前来看初始模型指令遵循能力还可以，需要进一步指引其输出更准确的<cmd></cmd>内容。

