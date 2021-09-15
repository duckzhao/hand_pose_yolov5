# hand_pose_yolov5
使用yolov5检测手势

手势数据集：https://gas.graviti.cn/dataset/datawhale/HandPose
点击右上角的探索数据集，即可下载该数据集，托管在国内的服务器上的，下载的很快。

训练好的权重文件：https://pan.baidu.com/s/1AYmZmr5K2f5fGnizjdyZ4w 提取码：68vu
## 使用方法
使用fastapi构建了一个web接口，可以将模型部署在服务器，前端使用http协议访问。
1. 部署
   1. 修改interface_of_model.py中的 weights 变量地址为本地的权重文件路径
   2. 确认本机已经配置了yolov5所必须的环境，https://github.com/ultralytics/yolov5/blob/master/requirements.txt
   3. 确认已经安装了fastapi和uvicorn两个用于构建接口的第三方库
   4. 运行interface_of_model文件即可
2. 测试
   1. test_interface文件为测试用例，使用摄像头时时捕获手势并送去服务器检测，使用前请确保机器中装有摄像头
   2. 修改detect函数中的 post地址'192.168.0.101'为服务器所在地址，可以使用ipconfig命令查看服务器地址，如果使用同一台机器启动该项目，则地址可改为'127.0.0.1'或者'0.0.0.0'。
   3. 本机仅需opencv和requests两个环境即可
   4. 运行interface_of_model文件开始测试
3. 自己训练
   1. 请前去yolov5官网学习训练自己数据的方法
   2. 本项目中的convert_ori_dataset_to_yolo.py文件可帮助你快速将hand_pose数据集转换为yolo的训练格式
## 最后
本项目使用yolov5s model构建，训练的速度十分快，测试准确率也很高。感兴趣的朋友可以https://github.com/ultralytics/yolov5 去官网查看更多教程，如果想对视频文件或者单帧图片进行测试，可以使用yolov5项目自带的detect文件进行测试。