import torch
# from torchvision.models import AlexNet
from torchviz import make_dot
import model.rca_pic as p
import tensorwatch as tw

model = p.make_model()
 


# x=torch.rand(8,64,32,32)

# y=model(x)


# g = make_dot(y,params=dict(list(model.named_parameters())))

# g.render('espnet_model', view=False)

 
 
# import torch
# import tensorwatch as tw
# # from lanenet_model.blocks import ESPNet_Encoder # 这是我自己定义的一个网络
 
# # 其实就两句话
# model = p.make_model()
# img = tw.draw_model(model, [1, 64, 32, 32])
# # print(img)
# print(dir(img))
# img.save(r'./a.png')
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),      #(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
 
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 
dummy_input = torch.rand(13, 1, 28, 28) #假设输入13张1*28*28的图片
model = LeNet()
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))