Assignment-4
TEAM MEMBER NAMES(size=1):
Name:Gadiraju Sanjay Varma
Email:18pa1a1211@vishnu.edu.in

Process of developing model:
    Here we are developing a model of getting 99.4% accuracy.
    Initially the size is of 28x28 pixel.
    So as the size is very less we can do two convolution blocks to obtain good accuracy
    Here I started with 32 kernals  and continued the 32 kernlas for the first convolution block
    This type of method is mostly followed by state of art networks
    Next i used maxpooling followed by 1x1 kernal to decrease my no of channels from 32 to 16
    After i developed the next convolution block with using 16 kernals.
    It is actually beneficial for using kernals in multiple of 8
    So my primary aim is to develop a model of less than 20,000 parameters
    I developed that and primary goal is completed
    Next is to improve accuracy step by step
    Next step is i used the batch normalisation which leads to overfitting
    SO i used dropout with probability of 5 percent and i overcome this overfitting
    Next i used Adaptive global average pooling which takes 4x4 and convert it into 1x1 of 10 channels
    So i got 99.43% accuracy.
    Thank you for reading
