# Self-Guided-Network
This is an unofficial reimplement of "Self-Guided Network for Fast Image Denoising" of ICCV2019.

The SGN Network is implemented by class of SelfGuidedNet in script "modules/featNet.py" .

The whole framework is a reimplement of paper "ATTENTION MECHANISM ENHANCED KERNEL PREDICTION NETWORKS FOR DENOISING OF BURST IMAGES" and replace the feature extraction network U-net by SGN.

## How to train the network
First check out the "param.cfg" and repalce params for your self.

Then, run the command "python main.py -c param.cfg".
