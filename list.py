import os
from tqdm import tqdm

if __name__ == '__main__':
    rootdir = 'E:\experiments\wheat-detection\data_field\pest3'
    rootimagesdir = os.path.join(rootdir, 'images')
    rootlabelsdir = os.path.join(rootdir, 'labels')
    assert os.path.exists(rootimagesdir), f'{rootimagesdir} not exists.'
    assert os.path.exists(rootlabelsdir), f'{rootlabelsdir} not exists.'

    imagesets = ['train', 'val', 'test']    # 将images/train val test中的图像的绝对地址分别写入txt中

    for sets in imagesets:
        imagesdir = os.path.join(rootimagesdir, sets)
        with open(os.path.join(rootdir, f'{sets}.txt'), 'w') as f:
            filelist = os.listdir(imagesdir)
            for images in tqdm(filelist):
                f.write(os.path.join(imagesdir, images) + '\n')
