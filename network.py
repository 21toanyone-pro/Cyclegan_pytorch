import torch.nn as nn
import torch



def define_G(input_nc, output_nc, netG, n_layers_G = 3, norm='batch', init_type='normal'):
    if netG == 'Resnet':
        net = Generators(in_channels=3,out_channels=3)
    elif netG =='Projetion':
        net = Projection_Generators(in_channels=3,out_channels=3)
    return net

def define_D(input_nc, output_nc, netD, n_layers_D = 3, norm='batch', init_type='normal'):
    if netD == 'basic':
        net = Discriminators(in_channels=3)
    elif netD =='Projetion':
        net = Projection_Discriminators(in_channels=3)
    return net

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block =[]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding = 0, bias= use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block +=[nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),norm_layer(dim)]

        self.conv_block=nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Projection_Generators(nn.Module):
    def __init__(self, in_channels, out_channels, nker = 64, norm_layer=nn.InstanceNorm2d, use_dropout = False, n_blocks =6,padding_type = 'reflect'):
        super(Generators, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, nker, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(nker), nn.ReLU(True)]
        n_downsampling = 2
        # downsampling layer (input image 피쳐 추출)

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ nn.Conv2d(nker *mult, nker * mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(nker * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        
        #resnet Block (정보손실 적고, 고해상도 처리 가능)
        for i in range(n_blocks):
            model += [ResnetBlock(nker * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        #upsampling layers (image style translation)
        for i in range(n_downsampling):
            mult = 2 ** ( n_downsampling - i)
            model += [nn.ConvTranspose2d(nker * mult, int(nker * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1,
            bias=use_bias), norm_layer(int(nker * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(nker, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Generators(nn.Module):
    def __init__(self, in_channels, out_channels, nker = 64, norm_layer=nn.InstanceNorm2d, use_dropout = False, n_blocks =6,padding_type = 'reflect'):
        super(Generators, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d
        #skip-connection
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, nker, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(nker), nn.ReLU(True)]

        n_downsampling = 2
        # downsampling layer
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ nn.Conv2d(nker *mult, nker * mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(nker * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        
        #resnet Block 9
        for i in range(n_blocks):
            model += [ResnetBlock(nker * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        #upsampling layers (image style translation)
        for i in range(n_downsampling):
            mult = 2 ** ( n_downsampling - i)
            model += [nn.ConvTranspose2d(nker * mult, nker * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1,
            bias=use_bias), norm_layer(int(nker * mult / 2)), nn.ReLU(True)]
    
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(nker, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminators(nn.Module):
    def __init__(self, in_channels, n_layers=3,nker = 64, norm_layer=nn.BatchNorm2d):
        super(Discriminators, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        #3 64
        sequence = [nn.Conv2d(in_channels, nker, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
       
        nf_mult =1
        nf_mult_prev = 1
        #64 128 256
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 **n, 8)
            sequence += [nn.Conv2d(nker *nf_mult_prev, nker * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(nker * nf_mult), nn.LeakyReLU(0.2, True)]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        #256 512
        sequence +=[nn.Conv2d(nker *nf_mult_prev, nker * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(nker * nf_mult), nn.LeakyReLU(0.2, True)]
            
        #(512, 1) kernel = 4
        sequence +=[nn.Conv2d(nker*nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)
        # kernel size 4 <- DCGAN 
    def forward(self,x):
        return self.model(x)

class Projection_Discriminators(nn.Module):
    def __init__(self, in_channels, n_layers=3,nker = 64, num_classes= 68, norm_layer=nn.BatchNorm2d):
        super(Discriminators, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        sequence = [nn.Conv2d(in_channels, nker, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
       
        nf_mult =1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 **n, 8)
            sequence += [nn.Conv2d(nker *nf_mult_prev, nker * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(nker * nf_mult), nn.LeakyReLU(0.2, True)]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence +=[nn.Conv2d(nker *nf_mult_prev, nker * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(nker * nf_mult), nn.LeakyReLU(0.2, True)]


        self.linears = nn.Linear(512,1)
        self.L_y = nn.Embedding(num_classes, 512, max_norm=1)
        #(512, 1) kernel = 4
        #sequence +=[nn.Conv2d(nker*nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)
        # kernel size 4 <- DCGAN 
    def forward(self,x,y):
        h = self.model(x)
        h = torch.sum(h, dim=(2,3))
        w_y = self.L_y(y)
        
        return self.linears(h) + torch.sum(w_y*h, dim=1, keepdim=True)
