# 3D Pose Motion Representation for Action Recognition

This repo follows the course 3D-Vision at ETH Zurich. Our project is 3D Pose Motion Representation for Action Recognition.

Group members include: Shengyu Huang, Ye Hong, Jingtong Li. 

Supervisors: Bugra Tekin, Federica Bogo, Taein Kwon

## Document
The folder ``docs`` contains all the documents throughout the whole semester. ``Final_Report.pdf`` is our final submitted report. There are some visualizations in  ``scripts\eval.ipynb``. You may not run it as all data files are ignored.

## Experiment
We only adopt open-sourced codes in pose estimation parts. We implement all other parts from scratch. 

### Pose estimation
The folder ``pose-hg-3d`` is cloned from [here](https://github.com/xingyizhou/pose-hg-3d). It was implemented in Torch 7. This [link](https://www.evernote.com/l/AnSiRotWeItD4b1R0MFXveYSgTuM3oPirkg) provides comprehensive instructions on configuring the required environment. We modify the file ``demo.lua`` to process all the cropped frames in Penn Action dataset. You have to first download Penn Action dataset and crop all the frames using scripts ``crop_frames.py`` under the folder ``src\preprocess``. 

### PoTion representation
With 3D pose estimation, we use python files under the folder ``src\preprocess`` to obtain PoTion representations. Scripts to obtain 2D PoTion, 3D PoTion as well as multi-view 2D PoTion are all provided and could be guessed from the file names. You have to modify ``config.py`` to fit your file paths. We use multiprocessing tool to speed up generating PoTion representations. The codes can run under both MacOS and Ubuntu OS. 

### Action recognition
With PoTion representations, we use CNN model to recognize actions. The folder ``src\2d`` contains the codes for 2D CNN part. The folder ``src\3d`` contains the codes for 3D CNN part. The folder ``src\meta`` contains meta data for splitting training, validation, and test samples. Again, you have to modify ``configs.py`` to reproduce the results. 
