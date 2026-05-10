# **SurRoL**

## **1.15**

### 一、

需要告诉系统：“这个 Python 脚本必须用 NVIDIA 显卡跑”。

**方法 A：使用环境变量（推荐）** 在运行 Python 命令前加上 NVIDIA 的专用环境变量：

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python test_multiple_scenes_keyboard.py



### 二、

 (base) zjj@zjj:~/TouchDriver_2024_09_19$ ls -l bin/ 

 drwxrwxr-x 2 zjj zjj  4096 9月 20 2024 fonts 

 -rw-rw-r-- 1 zjj zjj   245 9月 20 2024 imgui.ini 

 -rwxrwxr-x 1 zjj zjj 110776 11月 16 2023 Touch_AdvancedConfig   修改配置（重命名right）

 -rwxrwxr-x 1 zjj zjj 7781240 9月 20 2024 TouchCheckup 图形化诊断工具

 -rwxrwxr-x 1 zjj zjj 122560 11月 16 2023 Touch_HeadlessSetup  重置配置

SurRoL 的教程是基于老版本驱动写的，所以还在叫 `Touch_Setup`，但在你的新版驱动里，这些工具已经被**替换**了。

根据你列出的文件列表，请按照以下对应关系操作：

1. 替代 Touch_Diagnostic（诊断工具）

请运行 **`TouchCheckup`**。 这是一个很大的文件（7.7MB），它就是新版的图形化诊断工具。

- **运行命令**：

  \# 确保 USB 已连接      **sudo ./bin/TouchCheckup**

运行图形化配置工具：

**sudo ./bin/Touch_AdvancedConfig**

​      把 **Device Name** 改回right点击 **Save**。退出软件。

### 三、

To start the SurRoL-v2 GUI with Touch (haptic device) support, run the following command:
要启动支持 Touch（触觉设备）的 SurRoL-v2 图形界面，请执行以下命令：

```
cd ./tests/
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python test_multiple_scenes_touch.py
```