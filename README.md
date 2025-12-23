## install
```bash
cd VITRA
git submodule update --init --recursive

# 适配cuda12.4进行的修改，原版是cuda11.7
cd thirdparty/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch
# 检查patch
git apply --check ../../../../../../patch/0001-fix-type-to-scalar_type.patch
# apply patch，没有输出说明正常
git am ../../../../../../patch/0001-fix-type-to-scalar_type.patch
# 进入目录thirdparty/HaWoR
cd ../../../../
# 检查patch
git apply --check ../../patch/0001-fix-cuda-version-to-12.4.patch
# apply patch
git am ../../patch/0001-fix-cuda-version-to-12.4.patch

conda create --name vitra python=3.10
# 根据cuda版本安装适配的pytorch，这里使用cuda12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# build pytorch3d 需要很久
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
pip install --no-build-isolation "chumpy@git+https://github.com/mattloper/chumpy"
conda install -c conda-forge ffmpeg

cd VITRA
pip install -e .
pip install -e ".[visulization]" --no-build-isolation

cd thirdparty/HaWoR
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0

# install droid-slam 需要很久
cd thirdparty/HaWoR/thirdparty/DROID-SLAM
python setup.py install

# 安装机器人逆运动学求解需要的包
conda install -c conda-forge pinocchio
pip install casadi meshcat
pip install logging_mp
```

## download

根据下列3个仓库的readme下载相关权重文件
- https://github.com/microsoft/VITRA
- https://github.com/ThunderVVV/HaWoR
- https://github.com/ThunderVVV/HaWoR/tree/main/thirdparty/DROID-SLAM


## img+prompt推理action
```bash
python scripts/inference_human_prediction.py \
    --config VITRA-VLA/VITRA-VLA-3B \
    --image_path ./examples/my.jpg \
    --sample_times 1 \
    --save_state_local \
    --use_right \
    --video_path ./example_human_my.mp4 \
    --mano_path ./weights/mano \
    --instruction "Left hand: None. Right hand: Pick up the drink on the table."
```

## 自己采的视频估计mano
```bash
cd thirdparty/HaWoR
python demo.py --video_path ./example/video_0.mp4 --vis_mode world --no_display
```

## mamo转化为xhand
```bash
python scripts/human_video_to_mano.py
```

## 可视化xhand
```bash
# 简化可视化（推荐先试这个）
python scripts/visualize_xhand_action.py --simple_vis

# 使用 G1 机器人模型可视化（需要 G1 模块）
python scripts/visualize_xhand_action.py --transform_to_robot --use_finger_mapping

# G1 机器人模型可视化并录制视频
python scripts/visualize_xhand_action.py \
  --transform_to_robot \
  --use_finger_mapping \
  --save_video \
  --output_video examples/robot_motion.mp4 \
  --fps 10

# 指定特定帧范围
python scripts/visualize_xhand_action.py --simple_vis --start_frame 0 --end_frame 50

# 调整播放速度
python scripts/visualize_xhand_action.py --fps 15
```