import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

    
class MyInception_v3(nn.Module):
    def __init__(self,transform_input=False,pretrained=False):
        super(MyInception_v3,self).__init__()
        self.transform_input=transform_input
        inception=models.inception_v3(pretrained=pretrained,init_weights=False)
        
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        
    def forward(self,x):
        outputs=[]
        
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3  #[15, 3, 480, 720]->[15, 3, 540, 960]
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32 #[15, 32, 239, 359]
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32 #[15, 3, 480, 720]
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64 #[15, 32, 237, 357]
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64 #[15, 64, 118, 178]
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80 #[15, 64, 118, 178]
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192 #[15, 192, 116, 176]
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192 #[15, 192, 57, 87]
        x = self.Mixed_5b(x)
        # 35 x 35 x 256 #[15, 256, 57, 87]
        x = self.Mixed_5c(x)
        # 35 x 35 x 288 #[15, 288, 57, 87]
        x = self.Mixed_5d(x)
        # 35 x 35 x 288 #[15, 288, 57, 87]->[15, 288, 65, 117]
        outputs.append(x) #append to the output
        
        x = self.Mixed_6a(x)
        # 17 x 17 x 768 #[15, 768, 28, 43]
        x = self.Mixed_6b(x)
        # 17 x 17 x 768 #[15, 768, 28, 43]
        x = self.Mixed_6c(x)
        # 17 x 17 x 768 #[15, 768, 28, 43]
        x = self.Mixed_6d(x)
        # 17 x 17 x 768 #[15, 768, 28, 43]
        x = self.Mixed_6e(x)
        # 17 x 17 x 768 #[15, 768, 28, 43]->[15, 768, 32, 58]
        outputs.append(x) # append to the output
        
        return outputs
    

class MyVGG16(nn.Module):
    def __init__(self,pretrained=False):
        super(MyVGG16,self).__init__()
        
        vgg=models.vgg16(pretrained=pretrained)
     
        self.features=vgg.features
        
    def forward(self,x):
        x=self.features(x)
        return [x]
    
    
class MyVGG19(nn.Module):
    def __init__(self,pretrained=False):
        super(MyVGG19,self).__init__()
        
        vgg=models.vgg19(pretrained=pretrained)
     
        self.features=vgg.features
        
    def forward(self,x):
        x=self.features(x)
        return [x]

class MyResNet50(nn.Module):

    def __init__(self,pretrained=False):
        super(MyResNet50,self).__init__()
        resnet=models.resnet50(pretrained=pretrained)

        self.conv1=resnet.conv1
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4




    def forward(self, x):
        outputs=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#(N, 64, 56, 56)

        x = self.layer1(x)#(N, 256, 56, 56)
        x = self.layer2(x)#(N, 512, 28, 28)
        x = self.layer3(x)#(N, 1024, 14, 14)
        x = self.layer4(x)#(N, 2048, 7, 7)
        outputs.append(x)

        return outputs

def get_backbone(name,pretrained=True):
    if name=="resnet":
        return MyResNet50(pretrained=pretrained)
    elif name=="vgg19":
        return MyVGG19(pretrained=pretrained)
    elif name=="inception":
        return MyInception_v3(pretrained=pretrained)
    pass

if __name__=="__main__":
    a=torch.randn((1,3,720,1080))
    m=get_backbone("resnet",pretrained=False)
    m(a)