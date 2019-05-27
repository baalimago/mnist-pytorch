# Mnist in pytorch

I did the mnist_linear, the mnist_cnn is the pytorch example for mnist (which uses a much more effective convolutional neural network, found here: https://github.com/pytorch/examples/tree/master/mnist).

The processed data for the cnn and linear are different, which is why they're saved differently and at different places.

Workers for the linear network do work, but it's not very efficient since most of the time is loading data to and from the GPU instead of actually doing calculations. 

I wanted to figure out the least amount of properties which achieves the highest accuracy, and it lands fairly consistently around 90% or so in the current configuration. I found that a lot of it depends on the weight initialization and couldn't push the current rather brute linear model to higher than ~92%. 

Every linear run is saved in the /results folder as csv if for further inspection. I went out and just left it running sometimes when i did large tests of parameters. Play around at your own leisure with test_setting() to see what works and what doesn't, it's quite informative!