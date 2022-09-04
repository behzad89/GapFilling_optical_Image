# Gapfilling of Optical Images (Sentinel-2)

![image](https://user-images.githubusercontent.com/47195556/188323034-13e05d63-d3d1-410f-b281-51c2d394b0cf.png)
![image](https://user-images.githubusercontent.com/47195556/188323048-ef1aa629-e063-4dda-acf6-7d5c4dd3b928.png)


A well-known issue impacting optical imagery is the presence of clouds. The need for cloud-free images at precise date is required in many operational monitoring applications. On the other hand, the SAR sensors are cloud-insensitive and they provide orthogonal information with respect to optical satellites
that enable the retrieval of information, which could be lost in optical images because of cloud cover. To alleviate this issue, a common approach consists
of interpolating temporally close optical images to approximate the missing contents of the target image, but this method uses only optical images,
and is not designed to exploit the SAR information. Other machine learning– based approaches such as dictionary learning [29, 30], and more recently deep
learning , have emerged to restore cloud-impacted optical images with SAR images. For instance in, optical and SAR time series are jointly processed with a CNN to retrieve the Normalized Difference Vegetation Index of the missing optical image.

Source: Deep Learning for Remote Sensing Images with Open Source Software
