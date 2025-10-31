# terminalRL

----
## 快速执行

----
## 数据集选择
采用github上开源项目https://github.com/Danau5tin/terminal-bench-rl，数据在https://github.com/Danau5tin/terminal-bench-rl/blob/main/dataset/latest_verified.csv。

----
## 整体思路
### 目标确定
- 训练：基于AReal构建一个基于终端指令的AI agent。
- 评测：基于terminal-bench测试集进行评测。
### 训练细节
1. 首个版本因为时间较紧，设计的reward func只看stderr和stdout输出，并给出二值奖励。
2. 流程为：prompt构建->轨迹收集->reward计算->反向更新。

## 遇到的问题：
1. ASearcher/train/asearcher.py::132每一次轨迹收集中，每个轮次的奖励似乎都会被下一轮覆盖，中间奖励似乎没有作用到最终训练信号中？是不需要吗，只需要最终一轮和ground truth作对比？