import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_gradcam import GradCAM, show_cam_on_image, center_crop_img
from model_ECA import efficientnetv2_s
# from model import efficientnetv2_s


def main():
    # ----载入自己的模型，按照自己训练集训练的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = efficientnetv2_s(num_classes=9)  # .to(device)
    # load model weights
    # model_weight_path = "./weights/model-196.pth"#se
    model_weight_path = "./model-199.pth"#ECA
    # model_weight_path = "./weights_ECA_grad/model-195.pth"  # ECA
    load = torch.load(model_weight_path, map_location=device)
    # print(load)
    load.pop('head.classifier.weight')
    load.pop('head.classifier.bias')
    model.load_state_dict(load,strict=False)  # 加载权重
    model.eval()
    target_layers = [model.blocks[-1]]  # 拿到最后一个层结构

    # 载入模型，模型权重---载入的模型是按照ImageNet训练的（不是自己的模型是从torchvision中导入的）
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #load image
    # img_path = "23602.jpg"
    img_path = "23939.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/0.jpg")
    plt.show()

    # load image 1
    # img_path = "25737.jpg"
    img_path = "25814.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/1.jpg")
    plt.show()

 # load image 2
    img_path = "22521.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 2  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/2.jpg")
    plt.show()

# load image 3
    img_path = "23116.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 3  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/3.jpg")
    plt.show()

# load image 4
    img_path = "24846.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 4  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/4.jpg")
    plt.show()

# load image 5
    img_path = "24606.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 5  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/5.jpg")
    plt.show()

# load image 6
    img_path = "24424.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 6  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/6.jpg")
    plt.show()

# load image 7
#     img_path = "25206.jpg"
    img_path = "25226.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 7  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/7.jpg")
    plt.show()

# load image 8
    img_path = "25550.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 感兴趣的label
    target_category = 8  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.savefig("grad_cam_pic0525/8.jpg")
    plt.show()

if __name__ == '__main__':
    main()
