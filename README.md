# Trained Ternary Quantization (TTQ)
TensorFlow implementation of paper:

[Trained Ternary Quantization](https://arxiv.org/pdf/1612.01064v1), by Zhu et al.

This implementation is based on [tensorpack](https://github.com/ppwwyyxx/tensorpack). Thanks to this framework which made this implementation extremely easy.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.8
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies, you can install later if needed)
```
+ Use [tcmalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) whenever possible
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```

## Usage

+ To train ResNet on CIFAR10 with fixed threshold:
```
cd examples/Ternary-Net/
python ./tw-cifar10-resnet.py --gpu 0,1 [--load MODEL_PATH] [--t threshold] [--n NSIZE]
```
**Note: We used 2 GPUs for training and pretrained model can be obtained using /examples/ResNet/**

+ To train ResNet on CIFAR10 with fixed sparsity:
```
cd examples/Ternary-Net/
python ./p-cifar10-resnet.py --gpu 0,1 [--load MODEL_PATH] [—p sparsity] [--n NSIZE]
```
+ To train AlexNet on ImageNet with fiexed threshold:
```
cd examples/Ternary-Net/
python ./tw-imagenet-alexnet.py --gpu 0,1,2,3 --data IMAGENET_PATH [--t threshold]
```
**Note: We used 4 GPUs for training**

## Logs
Some training logs can be found [here](./examples/Ternary-Net/train_log).

## Support

Please use [github issues](https://github.com/czhu95/ternarynet/issues) for any issues related to the code.
Send email to the authors for general questions related to the paper.

## Citation

If you use our code or models in your research, please cite:
```
@article{zhu2016trained,
  title={Trained Ternary Quantization},
  author={Zhu, Chenzhuo and Han, Song and Mao, Huizi and Dally, William J},
  journal={arXiv preprint arXiv:1612.01064},
  year={2016}
}
```
