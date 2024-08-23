<h1> ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection</h1>
<div style="text-align: center;">

[Arpit Bahety](https://arpitrf.github.io/), [Priyanka Mandikal](https://priyankamandikal.github.io/) [Ben Abbatematteo](https://babbatem.github.io/), [Roberto Martín-Martín](https://robertomartinmartin.com/) 

The University of Texas at Austin

[Project Page](https://robin-lab.cs.utexas.edu/ScrewMimic/) | [Arxiv](https://arxiv.org/abs/2405.03666) | [Video](https://www.youtube.com/watch?v=sPNoKgoxpuc)

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="assets/videos/teaser.m4v">

ScrewMimic aims to enable robots to learn bimanual manipulation behaviors from human video demonstrations and fine-tune them through interaction in the real world. Inspired by seminal work in psychology and biomechanics, we propose modeling the interaction between the two hands as a serial kinematic linkage — as a screw motion, in particular.

## Installation
You would need to setup Frankmocap (with the ego-centric model enabled). Refer to https://github.com/facebookresearch/frankmocap
```
git clone --single-branch --branch main https://github.com/UT-Austin-RobIn/ScrewMimic.git
cd ScrewMimic
conda create --name screwmimic python==3.10
conda activate screwmimic
pip install -r requirements.txt
```

## Usage
Run frankomcap on the human video
```
python -m demo.demo_handmocap --input_path {path_to_rgb_img_folder} --out_dir ./mocap_output/{folder_name}/ --view_type ego_centric --save_pred_pkl
```

Extract hand poses (remember to set the frankmocap output directory in the code)
```
python perception/extract_hand_poses.py --folder_name data/bottle_1
```

Obtain screw axis
```
python perception/extract_screw_action.py --f_name data/bottle_1 --hand right
```