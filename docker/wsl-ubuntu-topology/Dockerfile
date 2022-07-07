FROM pymesh/pymesh:py3.7

RUN apt-get update && apt install -y libgl1-mesa-glx
RUN apt-get install -y libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 \
    libxcb-xkb1 libxkbcommon-x11-0 libdbus-1-3
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install scikit-learn open3d vtk opencv-python shapely mayavi
