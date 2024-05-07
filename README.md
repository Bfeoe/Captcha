## 验证码识别课程作业练习

1.正常安装其他包。安装cv2和pytorch，终端执行<br>
~~~
pip install opencv-python
~~~

2.然后把外部库中的"site-packages/cv2/"中的"cv2.pyd"复制到"site-packages"，然后重启应用
（不执行步骤二应该问题也不大吧，整一整吧，也不麻烦，主要是cv2的函数库位置奇奇怪怪的）

3.执行顺序：Captcha_Data_Set.py  生成数据集
           pre.py               图像预处理执行两次（注释的读取路径和存储路径更改）
           main.py              训练模型
           Test.py              测试进行验证
