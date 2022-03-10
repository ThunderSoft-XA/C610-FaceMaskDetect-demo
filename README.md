# Qualcomm® QCS610 SoC Open Kit FaceMaskDetect-demo Developer documentation

## Introduce

This project relies on the QCS610 development kit, using the AI computing power and image processing capabilities of the development kit to collect images in real time, perform AI reasoning after preprocessing, and output the reasoning results. I use gstreamer and TFLITE to complete the above functions. When the pipeline obtains valid image data, it will automatically collect the image, perform AI inference, and return the position information for the facemask in the image frame.

At present, the global epidemic is still spreading. In order to ensure people's safe travel, people should be urged to wear masks to avoid the spread of the virus, so I want to use this demo to detect face masks to achieve this purpose. 

The project was built in x86 host with across complier tool and has been tested in Qualcomm® QCS610 Open Kits.

<center>Qualcomm® QCS610 SoC Open Kits</center>

![QCS610](./res/QCS610.png)

## Materials and Tools used for the project

1. Hardware materials

Except for the Open Kits,The following hardware materials are also needed:

* Type-C usb line

using the usb line to develop on Qualcomm® QCS610 SoC Open Kits.

![usb line](./res/usb.png )

* Charger

Direct power supply for Qualcomm® QCS610 SoC Open Kits.

![charger](./res/charger.jpg )

* DP-Line

using the universal dp line to connect LED displayer to  Qualcomm® QCS610 SoC Open Kits.

![DP-Line](./res/dpline.jpg)

* LCD-Displayer

using a LCD Displayer to display the FaceMaskDetect-demo interface from Qualcomm® QCS610 SoC Open Kits.

![LED-Displayer](./res/LED-Displayer.png)


## Environment configuration

This section mainly describes the source and configuration of some open source projects or third-party tools needed in the development process.

### Snapdragon Neural Processing Engine SDK

1. Download Snapdragon Neural Processing Engine SDK(version  : 1.37.0)
[https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools)
2. Enter include/zdl/ in Snapdragon Neural Processing Engine SDK root directory,copy all dir to demo project_root_dir/inference/snpe/

### Qualcomm FastCV

1. Download Qualcomm FastCV-Linux
[https://developer.qualcomm.com/software/fastcv-sdk](https://developer.qualcomm.com/software/fastcv-sdk)
2. Get fastcv.h header file and copy to project_root_dir/gst_pipe/

### OpenCV

The version of OpenCV is 3.4.3.There are many tutorials for opencv installation on the network.Therefore, you can completely refer to other people's installation methods for installation and configuration.For example:[https://www.cnblogs.com/uestc-mm/p/7338244.html](https://www.cnblogs.com/uestc-mm/p/7338244.html)


### google Protobuf

1. Download google protobuf (version  :  3.3)
[https://github.com/protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf)
2. Enter protobuf/src/google/ dir.Then,select all .h or .hpp .etc header file with maintain the structure of directory
3. copy header file dir to demo project_root_dir/pb_conf

### tensorflow 
1. Install [tensorflow](https://github.com/tensorflow/tensorflow) (version : 2.2.0)
2. Generate libtensorflow_lite.so and copy to demo project/inference/libs/tflite
3. copy tensorflow all *.h(C++ header file) to demo project/inference/tflite


## Compile

The compilation of the whole project is based on the yocto compilation tool, so you need to write some .bb and .conf files according to the specification. The link is [facemask_detect_0.1.bb](https://github.com/ThunderSoft-XA/C610-FaceMaskDetect-demo/blob/master/C610-FaceMaskDetect-demo/facemask_detect_0.1.bb) example.

Please refer to [the official Manual of Yocto](https://www.yoctoproject.org) for how to add layers,write layer.conf .Then,excute the command as follows:

```
bitbake facemask_detect
or
bitbake -b facemask_detect_0.1.bb
```

you will get a  executable bin file named FaceMaskDetect.Move it to the root of the source code.Next,push the whole project code to Qualcomm® QCS610 SoC device`s dir /data/.

```
adb root && adb disable-verity && adb reboot

adb root &&adb remount && adb shell mount -o remount,rw /

adb push xxx/xxx/sourcepath /data/

adb shell

cd data/FaceMaskDetect
```

## Configure and Usage

### 1. Configure

gstreamer pipeline google protobuf config:

| parameter| note |
| ---- | ---- | 
gstid : 0             |       //gstreamer pipeline  ID
gstname : "gst_zero"    |  //gstreamer pipeline name
sinkname : "gst_sink"  |  //gstreamer pipeline appsink Plug-in name
gsttype : CAMERA         |             // gstreamer pipeline type ,CAMERA/RTSP/LOCAL
path : 1     |                        // gstreamer pipeline source sink path ,rtsp  :  url,camera : camera id,local file  :  local video file path
decode : h264  |              // gstreamer pipeline decode type
framerate : 30 | 
format : NV12 |
width : 640 | 
height : 360 |
hw_dec : 1     |                //gstreamer pipeline hardware decode or software decode
need_calib : True  |             //image frame need calibration or not, 1  :  need

ai inference google protobuf config:
model_path : "/data/FaceMaskDetect/models/mask/face_mask_detection.tflite"    // model path 
runtime : GPU      // ai inference runtime : CPU/GPU/DSP/NNAPI/APU
data_format : NCHW    //ai inference data format NCHW/NHWC

### 2. Usage

This project only provides a simple command line interface.

Run FaceMaskDetect(This is the project name, which can be modified in the CMakeListe.txt file in the root directory of the project )：

Run the **FaceMaskDetect** test demo.This test demo has only command line output.

```
$ cd /data/<source root dir>
$ ./weston_dp_cmd helmet

Notice:
DP display uses weston graphical interface support, so I provide weston_dp_cmd script in the project root directory 

```
