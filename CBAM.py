import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ChannelAttention(nn.Module):   # ilk kısım
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),


            
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  # son kısım
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        channel_att_map = self.channel_attention(x) # önce channel attention
        x = x * channel_att_map    # element wise mul ile birleşiyior
         
        # Apply spatial attention
        spatial_att_map = self.spatial_attention(x)
        x = x * spatial_att_map
        
        return x


# Example usage with a sample CNN
class SampleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(SampleCNN, self).__init__()
        # Initial convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # CBAM after the first block
        self.cbam1 = CBAM(64)
        
        # Additional convolutional layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # CBAM after the second block
        self.cbam2 = CBAM(128)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"Cbam öncesi shape:{x.shape}")
        # Apply CBAM
        x = self.cbam1(x)
        
        print(f"Cbam sonrası shape:{x.shape}")

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Apply CBAM
        x = self.cbam2(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    


CBAM_model = SampleCNN(num_classes = 1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CBAM_model.to(device)

input_tensor = torch.rand(1, 3, 224, 224).to(device)  
with torch.no_grad():
    output = CBAM_model(input_tensor)
    print(output.shape) 

# CBAM BOYUT DĞEİŞİKLİĞİNE SEBEP OLMAZ BU YÜZDEN HER YERİN SONRASINA EKLENEBİLİR.

summary(CBAM_model, input_size=(3, 224, 224), device = str(device))

# 218000 parametre

# Cbam öncesi shape:torch.Size([2, 64, 56, 56]) # 
# Cbam sonrası shape:torch.Size([2, 64, 56, 56])
