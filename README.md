Project Description: Lane Detection using U-Net Architecture
This is an old project I made it implements a U-Net-based deep learning model for lane detection, a critical task in autonomous driving systems. 
The architecture is designed to segment lane markings in images by combining multi-scale feature extraction with precise localization.

Key Architectural Components:

Contracting Path (Encoder):

A series of conv_blocks progressively downsample the input using 3x3 convolutions (ReLU-activated, He-initialized) and 2x2 max-pooling, doubling filters at each step (32 → 64 → 128 → 256).

Dropout layers (prob=0.3) in deeper layers (e.g., 256 and 512 filters) mitigate overfitting.

Skip connections preserve high-resolution features from each block for later fusion.

Expanding Path (Decoder):

Transposed convolutions (Conv2DTranspose) upsample features, followed by concatenation with skip connections from the encoder to recover spatial details.

Each upsampling_block applies two 3x3 convolutions to refine merged features, halving filters at each step (128 → 64 → 32).

Final Layers:

A 1x1 convolution maps the final decoder output to n_classes, producing a segmentation mask (e.g., binary lane vs. background).
