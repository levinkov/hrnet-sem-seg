# High-resolution networks (HRNets) for Semantic Segmentation
**Disclaimer:** I do not own any copyright for this implementation.
I simply ripped out the network code and packaged it nicely, so that anyone can copy the code into their project and use it in their own training procedure.

This repo is based on commit `88419ab18813f2c9193985e2d4d31d3d07abe839` from the original [repo](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).
I only made it possible to specify the number of input and output channels as function parameters.

This code requires Python 3.7+, PyTorch 1.1+, and all the usual packages. The only non-conventional dependency is [yacs](https://pypi.org/project/yacs/).

# Usage
...is very simple! If you put the network code in `${PROJECT_DIR}/model/`, then:
```python
from model.hrnet import get_hrnet

model = get_hrnet(config_name, num_classes, num_input_channels)
```
and you're good to go!

#### Parameters description:
* `config_name` can be one of the following: `'small_v1'`, `'small_v2'`, `'big'`.
The difference is only in the number of trainable parameters, which is, correspondingly, 1.5M, 3.9M, and 65.8M.
* `num_classes` is number of predicted semantic classes, i.e. number of output channels.
* `num_input_channels` is number of input channels. For example, one can train on RGB (`num_input_channels=3`) or gray-scale (`num_input_channels=1`) images.

## Citation
If you find this code helpful in your research or work, please, cite the original papers:
```
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```
And, maybe, acknowledge this repo :)
