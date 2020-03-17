from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4modeltrainer import ModelTrainer

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]

    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())

      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None

class QuizNet(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(QuizNet, self).__init__(name)

        
        self.conv0= self.create_conv2d(3, 32, dropout=dropout_value,kernel_size=(1,1),padding=0) 
        self.conv1 = self.create_conv2d(32, 32, dropout=dropout_value) 
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = self.create_conv2d(32, 64, dropout=dropout_value,kernel_size=(1,1),padding=0)
        self.conv4 = self.create_conv2d(64, 64, dropout=dropout_value) 
        self.conv5 = self.create_conv2d(64, 64, dropout=dropout_value) 

        
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.conv6 = self.create_conv2d(64, 128, dilation=1, padding=0,kernel_size=(1,1)) 
        self.conv7 = self.create_conv2d(128, 128, dropout=dropout_value) 
        self.conv8 = self.create_conv2d(128, 256, dropout=dropout_value) 

        
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.conv8 = self.create_conv2d(128, 128, dropout=dropout_value) 
        self.conv9 = self.create_conv2d(128, 128, dropout=dropout_value) 

      
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x1 = self.conv0(x) 
        x2 = self.conv1(x1)
        x3 = self.conv2(x1+x2) 

        x4 = self.pool1(x1+x2+x3) 
        x4=self.conv3(x4)      

        x5 = self.conv4(x4)   
        x6 = self.conv5(x4+x5) 
        x7=  self.conv5(x4+x5+x6) 
        x8 = self.pool2(x5+x6+x7) 
        x8=self.conv6(x8)         
        x9 = self.conv7(x8)      
        x10 = self.conv8(x8+x9)   
        x11 = self.conv9(x8+x9+x10) 
        y = self.gap(x11)
        y = self.conv10(y)

        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)