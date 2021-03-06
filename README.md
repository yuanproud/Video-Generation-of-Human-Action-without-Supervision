GAN-Based Video Generation of Human Action by Using Two Frames
==========

A PyTorch implementation of "GAN-Based Video Generation of Human Action by Using Two Frames"<br>

Requirements
----------
* PyTorch
* pytorchvision
* numpy
* imageio
* scipy
* spacepy

Dataset Preparation
----------
1. To reproduce our results, download the [Huamn 3.6M dataset](https://vision.imar.ro/human3.6m/main_login.php).
2. Resize the videos into 256*256, and save all the videos in a single folder named Videos_"action"_256
3. Use the [Real-Time Multi-Person 2D Pose Estimation Using Part Affinity fields](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) pre-train model to extract the 15 human joint points, and save all the videos in a single folder named Pose_D2_256_"action".

Demo
----------
* Download [pre-train pytorch models](https://drive.google.com/drive/folders/1e57BfOmdBrufcy5au5FwqrkJ07KoWaJ0?usp=sharing)
* Run the test.py

Training
----------
<pre><code>python pose_complement.py
<br></br>
optional arguments:
--batch_size      training batch size [default value is 32]
--niter           epoch of training [default value is 5000]
--pre_train       whether use the pre-train model [default value is -1]
--imiage_size     training images size [default value is 256]
--action          Action of you want to generate [default value is 'Walking']</code></pre>
<pre><code>python sequence_complement.py 
<br></br>
optional arguments:
--batch_size      training batch size [default value is 2]
--niter           epoch of training [default value is 1000]
--pre_train       whether use the pre-train model [default value is -1]
--imiage_size     training images size [default value is 256]
--action          Action of you want to generate [default value is 'Walking']</code></pre>
<pre><code>python heatmap_to_RGB.py 
<br></br>
optional arguments:
--batch_size      training batch size [default value is 8]
--niter           epoch of training [default value is 3000]
--pre_train       whether use the pre-train model [default value is -1]
--imiage_size     training images size [default value is 256]
--action          Action of you want to generate [default value is 'Walking']</code></pre>
  
* All the training will get good performance until 500 epoch.
* The parameters will save in the different folder separately.

Performance Evaluation
----------
<img src="result/Performance Evaluation.png" width="500" height="300" style="float:middle;">

Visual Results Comparing
----------

### Sitting
Ours
<p float="left">
  <img src="result/fake_sitting_epoch-0.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_epoch-1.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_epoch-2.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_epoch-4.gif" width="180" height="180" style="float:middle;">
</p>

[1]<br>
<br>
<p float='left'>
  <img src="result/sitting_24.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_37.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_40.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_42.gif" width="180" height="180" style="float:middle;">
</p>

<img src="result/4.png" alt="overview" style="float:middle;">

### Walking
Ours
<p float="left">
  <img src="result/fake_walking_epoch-2.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_walking_epoch-3.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_walking_epoch-4.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_walking_epoch-5.gif" width="180" height="180" style="float:middle;">
</p>

[1]<br>
<br>
<p float='left'>
  <img src="result/walking_11.gif" width="180" height="180" style="float:middle;">
  <img src="result/walking_30.gif" width="180" height="180" style="float:middle;">
  <img src="result/walking_33.gif" width="180" height="180" style="float:middle;">
  <img src="result/walking_38.gif" width="180" height="180" style="float:middle;">
</p>

<img src="result/4.png" alt="overview" style="float:middle;">
### Sitting Down
Ours
<p float="left">
  <img src="result/fake_sitting_down_epoch-0.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_down_epoch-2.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_down_epoch-6.gif" width="180" height="180" style="float:middle;">
  <img src="result/fake_sitting_down_epoch-7.gif" width="180" height="180" style="float:middle;">
</p>

[1]<br>
<br>
<p float='left'>
  <img src="result/sitting_down_6.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_down_20.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_down_29.gif" width="180" height="180" style="float:middle;">
  <img src="result/sitting_down_39.gif" width="180" height="180" style="float:middle;">
</p>

<img src="result/4.png" alt="overview" style="float:middle;">
