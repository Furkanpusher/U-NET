import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module): # mimarinin her yerinde üst üste convolution işlemleri var o yüzden bunu önden tanımlayacağım

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3,3), stride = 1, padding = 1, bias = False),
                        # 3x3 lük 64 tane conv uygulucaz ve boyutu aynı tutmak için padding 1 uyguluyoruz
                        # ama orijinal makalede padding kullanılmamış o yüzden convolution sonrasında resim boyutu korunmamış
            nn.BatchNorm2d(num_features = out_channels), 
            # Şart değil fakat weight değerlerini kabul edilebilir bir ağırlıkta tutmaya yardımcı olur
            # Birde nöral networkler 0 ile 1 aralığındaki değerler ile Türevin daha sabit kalması ve aktivasyon fonksiyonlarının
            # bu aralıklarda daha iyi çalışmaları gibi sebeplerden dolayı daha iyi performans gösterirler.
            nn.ReLU(inplace = True) # başka fonksiyonlarda kullanılabilir
        )
    
    def forward(self, x):  # inputu modelden geçircek
        return self.conv(x)

class U_NET(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 3, features = [64,128,256,512]): 
                    # tüm model boyunca bu boyutlarla uğraşcağımız için işimizi kolaylaştırdık
        super(U_NET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = 2) #klassik pooling işlemi burda padding yok

        # U-net in Down(aşağı iniş) kısmı

        for feature in features:
            self.downs.append(DoubleConv(in_channels = in_channels, out_channels = feature)) # in = 3, out = 64 oldu
            in_channels = feature # in_channels = 64 oldu aslında in_channels = out_channels gibi bişi yaptık
        
        # U-net in Up(yukarı çıkış) kısmı
        # ve bu sefer tersten gidicez yani yüksek channels dan düşük channelse geçicez
        for feature in reversed(features): # ters çevirdik
            self.ups.append(nn.ConvTranspose2d(
                in_channels = feature * 2, out_channels = feature, kernel_size = (2,2), stride = 2
                # burası resmin boyutunu 2 katına çıkarıcak burda transpoz yöntemi yerine upscale da yapabilirdik 
                # ama ben bunu seçtim 
            ))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottle_neck = DoubleConv(features[-1], features[-1] * 2) # 512x1024 olcak ve burası en alt kısım

        self.last_conv = nn.Conv2d(features[0], out_channels = out_channels, kernel_size = (1,1))
        # burası en son conv işlemi ve tek bir conv olcağı için doubleconv u kullanmadık
 

    def forward(self, x): 
        skip_connections = [] # burda o atlamalı bağlantılar öncesindeki conv bilgilerini tutup aktarıcaz
        for down in self.downs: # her aşağı inmeden önce atlamalı bağlantı var  
            x = down(x) 
            skip_connections.append(x) # ekledik ve sonra pooling yapıcaz
            x = self.pool(x)  # pooling ile bir alta inip tekrardan eklicez 

        x = self.bottle_neck(x) # en alta geldik
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2): # ŞİMDİ BURDA SKİP_CONNECTİONSUN DİĞER UCUNA ULAŞICAZ
                                            # step i 2 kullandık çünkü yukarı çıkıp 2 conv yapıyoruz
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]  # Bu satır değiştirildi, doğrudan `skip_connections` değişkeni etkilenmesin

            
            if x.shape != skip_connection.shape:  # Birleştirilen 2 ucunda aynı boyutlarda olması gerekiyor. bu önemli 
                print('Boyutlar Uyuşmadı, boyutlar eşitleniyor...')
                x = TF.resize(img = x, size = skip_connection.shape[2:]) # sadece yükseklik ve genişlik için son 2 yi alcaz

            concat_skip = torch.cat((skip_connection, x), dim = 1)# burda bağlama işleminin kendisi yapılıyor
            x = self.ups[index +1](concat_skip)

        return self.last_conv(x)

def test(): 
    x = torch.randn(3, 3, 160,160) # batch_size = 3, channels = 3, 160x160 
    # burda mini batch kullanmadık o yüzden pek önemli değil ama ileride kullanacağımız için şimdiden öyle test edelim 
    # 3 channels seçtim çünkü resim BGR formatında yani 3 boyutlu olcak 

    model = U_NET(in_channels = 3, out_channels = 3) # 3 boyutlu bir resim alıp 3 boyutlu bir resim döndürücek
    # grayscale çıktı isteseydik buraya 1 koyardık 

    tahminler = model(x)
    print(f"Tahminlerimizin boyutu:{tahminler.shape}")
    print(f"Girdilerin boyutu:{x.shape}")

    assert tahminler.shape == x.shape

if __name__ == '__main__': 
    test()
# ÇIKTILAR
Tahminlerimizin boyutu:torch.Size([3, 3, 160, 160])
Girdilerin boyutu:torch.Size([3, 3, 160, 160])
