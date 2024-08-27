# '''
#   图片格式批量转换
# '''
#
#
# def istype(filetype):
#     '''
#     判断是否为图片文件
#     :param filetype: 文件扩展名
#     :return: 是，返回True，不是，返回False
#     '''
#     filetype = filetype.lower()  # 扩展名转换为小写
#     # 判断是否为图片格式
#     if filetype == '.jpg' or filetype == '.jpeg' or filetype == '.png' or filetype == '.gif' or filetype == '.bmp' or filetype == '.tif':
#         return True  # 返回True
#     else:
#         return False
#
#
# def thumpic(filepath, width, height):
#     try:
#
#         image = Image.open(filepath)  # 打开图片文件
#         # 按指定大小对图片进行缩放（不一定完全按照设置的宽度和高度，而是按比例缩放到最接近的大小）
#         image.thumbnail((width, height))
#         # 下面的方法会以严格按指定大小对图片进行缩放，但可能会失真
#         # image = image.resize((width, height))
#         image.save(filepath)  # 保存缩放后的图片
#         # print('图片处理完成……')
#         # os.startfile(path) # 打开指定路径进行预览
#     except Exception as e:
#         print(e)
#
#
# import os
# from PIL import Image
#
# while True:
#     oldpath = input('请输入要转换格式的图片路径：')
#     newpath = input('请输入转换格式后的图片保存路径：')
#     flag = int(input('''要转换的格式：
# 1、jpg  2、jpeg  3、png  4、gif  5、bmp  6、tif
# 请选择：'''))
#     width = int(input('请输入宽度限制：'))
#     height = int(input('请输入高度限制：'))
#     list = os.listdir(oldpath)  # 遍历选择的文件夹
#     for i in range(0, len(list)):  # 遍历文件列表
#         filepath = os.path.join(oldpath, list[i])  # 记录遍历到的文件名
#         if os.path.isfile(filepath):  # 判断是否为文件
#             filename = os.path.splitext(list[i])[0]  # 获取文件名
#             filetype = os.path.splitext(list[i])[1]  # 获取扩展名
#             if istype(filetype):  # 判断是否为图片文件
#                 img = Image.open(filepath)  # 打开图片文件
#                 # 根据选择的格式转换图片，并保存
#                 if flag == 1:
#                     img = img.convert('RGB')  # 将图片转换为RGB格式，因为jpg格式不支持透明度
#                     img.save(os.path.join(newpath, filename + '.jpg'), 'jpeg')
#                     thumpic(os.path.join(newpath, filename + '.jpg'), width, height)
#                 elif flag == 2:
#                     img = img.convert('RGB')
#                     img.save(os.path.join(newpath, filename + '.jpeg'), 'jpeg')
#                     thumpic(os.path.join(newpath, filename + '.jpeg'), width, height)
#                 elif flag == 3:
#                     img.save(os.path.join(newpath, filename + '.png'), 'png')
#                     thumpic(os.path.join(newpath, filename + '.png'), width, height)
#                 elif flag == 4:
#                     img.save(os.path.join(newpath, filename + '.gif'), 'gif')
#                     thumpic(os.path.join(newpath, filename + '.gif'), width, height)
#                 elif flag == 5:
#                     img.save(os.path.join(newpath, filename + '.bmp'), 'bmp')
#                     thumpic(os.path.join(newpath, filename + '.bmp'), width, height)
#                 elif flag == 6:
#                     img.save(os.path.join(newpath, filename + '.tif'), 'tiff')
#                     thumpic(os.path.join(newpath, filename + '.tif'), width, height)
#     os.startfile(newpath)
#
#     print('格式转换完成……')


# from PIL import Image
# import os
#
# # 原始文件夹路径
# original_folder = 'E:\experiments\wheat-detection\论文\提交\SR/figures/12'
# # 保存的新文件夹路径
# new_folder = 'E:\experiments\wheat-detection\论文\提交\SR/figures/122'
#
# # 遍历原始文件夹中的图像
# for filename in os.listdir(original_folder):
#     img = Image.open(os.path.join(original_folder, filename))
#     # 改变尺寸
#     img_resized = img.resize((800, 600))   #这里是你要转换的尺寸
#     # 保存到新文件夹
#     img_resized.save(os.path.join(new_folder, filename))


import os
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# 原始图片文件夹路径
folder_path = 'E:\experiments\wheat-detection\论文\提交\SR/figures/2'

# 新建保存调整大小后图片的文件夹路径
output_folder_path = 'E:\experiments\wheat-detection\论文\提交\SR/figures/22'
# os.makedirs(output_folder_path, exist_ok=True)

# 获取原始图片文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 遍历原始图片文件夹中的所有文件
for file_name in file_list:
    # 判断文件是否为 PNG 格式
    if file_name.endswith('.jpg'):
        # 读取图像
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)

        # 调整图像大小为 512x512
        resized_image = image.resize((800, 600))

        # 保存调整大小后的图像到新的文件夹中，保持图片名称不变
        output_path = os.path.join(output_folder_path, file_name)
        resized_image.save(output_path)

        print(f"Resized {file_name} to 512x512 and saved as {file_name} in the output folder.")

