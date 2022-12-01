# 说明
* 代码是大幅度参照的李沐老师, 如有错误请指正.本机运行测试无问题<br>
* 在笔记本3060上训练约30min<br>
* 运行环境是Pycharm,或者linux端命令运行,不需要参数<br>
* 需要改参数在env文件夹中的Env.py中修改

# 运行
* 运行 'MainTrain.py' 开始训练,画图有2种模式<br>
在代码line 14,15注释掉一个修改.<br>
loss绘图使用visdom绘图还是matplot绘图<br>
* 使用visdom画图前需要在cmd窗口或命令行输入"python -m visdom.server"<br>
打开画图后端,并在浏览器打开对应提示的网址 <br>
* 运行 MainTest_Manual.py 手动输入测试

# 文件结构
### dataset -数据集文件夹
- fra-eng: 下载的原始数据文件夹 <br>
- fra-eng-processed: 经过处理的数据文件 <br>

### env -参数环境文件夹
* 模型网络参数设置和大部分训练设置
### model -模型保存文件夹
* 保存训练模型的位置
### net -网络模型文件夹
* transformer模型
* 支持模型需要的层和方法
### util -工具类文件夹
* 画图工具<br>
* 累加器<br>
* 计时器<br>
* 数据处理和加载<br>
* 词元化-词库化处理<br>
* 李沐老师的d2l库
### MainTest_Manual.py
* 运行这个.手动输入测试
### MainTrain.py
* 运行这个.开始训练
### Predict.py
* 不运行这个.作为MainTest_Manual.py和MainTrain.py的运行支持

# 运行环境
* 处理器:8C14T &nbsp; 内存:16G &nbsp; 显存:6G <br>
* 依赖参照requirements文件
