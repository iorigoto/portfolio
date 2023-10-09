#1️⃣ 画像の読み込みと前処理
from PIL import Image
import torchvision.transforms as transforms
import torch

def load_image(image_path, max_size=None, shape=None):
    # ... (中略)
    if max_size:
        # リサイズするコード
        pass
    if shape:
        # shapeに合わせてリサイズするコード
        pass
    # ... (中略)
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

content_image_path = "317px-Nike_of_Samothrake_Louvre_Ma2369_n4.jpg"
style_image_path = "primitive-Logo_NIKE.png"


# Now 'image' is an RGB image, you can save or use it

content = load_image(content_image_path, max_size=400)
style = load_image(style_image_path, shape=[content.shape[2], content.shape[3]])


#2️⃣ 事前訓練されたCNN（VGG19）の使用
import torchvision.models as models
from torchvision.models import VGG19_Weights

vgg = models.vgg19(pretrained=True).features
#vgg = models.vgg19(pretrained=True,weights=VGG19_Weights.IMAGENET1K_V1).features

#3️⃣ スタイルとコンテンツの特徴量の抽出
def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features

target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003)

style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv4_2': 0.2, 'conv5_1': 0.1}



#4️⃣ スタイル変換
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

#---------------------------------------------------------------------------------------
print("now NO4")
for i in range(8):
    print("now in for"+str(i))
    target_features = get_features(target, vgg)
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    style_loss = 0
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    for layer in style_weights:
        print("now in for"+str(layer))
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = gram_matrix(style_features[layer])
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2] * target_feature.shape[3])
        print('style_loss:',style_loss)
        
    total_loss = content_loss + 1e6 * style_loss
    print('total_loss:',total_loss)    
    print('optimizer.zero_grad')    
    optimizer.zero_grad()
    total_loss.backward()
    print('optimizer.step')    
    optimizer.step()

    if i % 100 == 0:
        torch.save(target, f'target_epoch_{i}.pt')

    print('ok next for')    
#---------------------------------------------------------------------------------------

# Convert the 'target' tensor image back to PIL image
print("now5")
target_image = target.clone().squeeze(0)
target_image = target_image.mul(255).byte()
target_image = target_image.cpu().transpose(0, 2).transpose(0, 1).numpy()

# Convert to PIL image
print("now6")
from PIL import Image
target_image_pil = Image.fromarray(target_image, 'RGB')

# Display the image using matplotlib
print("now7")
import matplotlib.pyplot as plt
#plt.imshow(target_image_pil)
#plt.axis('off')
#plt.show()
#------------------------------------------------------------------------------
# 特定の層（例: 'conv1_1'）の特徴マップを取得
print("# 特定の層（例: 'conv1_1'）の特徴マップを取得")
target_features = get_features(target, vgg)
feature_to_visualize = target_features['conv5_1']

print('# テンソルの形状を確認（[1, channel, height, width]）')
print(feature_to_visualize.shape)

print('# 任意のチャンネル（例：0番目）を選び、2D画像として可視化')
channel_to_visualize = feature_to_visualize[0, 0, :, :].detach().numpy()

print('conv_5_1')
plt.imshow(channel_to_visualize, cmap='gray')
plt.show()


