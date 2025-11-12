import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade = sample['grade']
        if random.random() < 0.7:
            # 对除了前三个特征之外的其他特征添加噪声
            noise = np.random.normal(0, 0.1, len(radiomic) - 3)
            radiomic = radiomic[:3] + [r + n for r, n in zip(radiomic[3:], noise)]
        if random.random() < 0.5:
            image = np.flip(image, 0)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        if random.random() < 0.5:
            image = np.flip(image, 2)
        return {'image3d': image, 'radiomic': radiomic, 'idh': idh, 'grade': grade}


class RandomIntensityChange(object):
    def __init__(self, intensity_shift_range=(-0.3, 0.3), intensity_scale_range=(0.7, 1.3)):
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, sample):
        image = sample['image3d']
        shift_value = np.random.uniform(self.intensity_shift_range[0], self.intensity_shift_range[1])
        scale_value = np.random.uniform(self.intensity_scale_range[0], self.intensity_scale_range[1])
        image = image * scale_value + shift_value
        sample['image3d'] = image
        return sample


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade = sample['grade']
        angle = round(np.random.uniform(-50, 50), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        return {'image3d': image, 'radiomic': radiomic, 'idh': idh, 'grade': grade}

class guiyihua(object):
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade = sample['grade']

        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))

        # 对每个像素点进行归一化
        for i in range(4):
            image[:, :, :, i] = (image[:, :, :, i] - mean[i]) / std[i]

        return {'image3d': image, 'radiomic': radiomic, 'idh': idh, 'grade': grade}



class Pad(object):
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade = sample['grade']

        image = np.pad(image, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')

        return {'image3d': image, 'radiomic':radiomic,'idh':idh,'grade':grade}
    #(240,240,155)>(240,240,160)

class RandomNoiseAugmentor(object):
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade = sample['grade']
        noise_prob=0.5
        if np.random.rand() < noise_prob:
            noise_type = np.random.choice(['gaussian', 'salt_and_pepper'])
            if noise_type == 'gaussian':
                image=self.add_gaussian_noise(image)
                return {'image3d': image, 'radiomic':radiomic,'idh':idh,'grade':grade}
            elif noise_type == 'salt_and_pepper':
                image=self.add_salt_and_pepper_noise(image)
                return {'image3d': image, 'radiomic':radiomic,'idh':idh,'grade':grade}
        else:
            return {'image3d': image, 'radiomic':radiomic,'idh':idh,'grade':grade}

    def add_gaussian_noise(self, image):
        mean = 0
        noise_strength=random.uniform(0, 0.3)
        var = noise_strength ** 2
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, -2, 2)
        return noisy_image.astype(np.float32)

    def add_salt_and_pepper_noise(self, image):
        p = random.uniform(0, 0.3)
        noisy_image = np.copy(image)
        salt = np.random.choice([0, 1], size=image.shape, p=[1 - p, p])
        pepper = np.random.choice([0, 1], size=image.shape, p=[1 - p, p])
        noisy_image[salt == 1] = 1
        noisy_image[pepper == 1] = -1
        return noisy_image.astype(np.float32)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image3d']
        radiomic = sample['radiomic']
        idh = sample['idh']
        grade=sample['grade']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = torch.from_numpy(image.copy()).float()

        return {'image3d': image, 'radiomic':radiomic,'idh':idh,'grade':grade}


class RandomCropAndPad(object):
    def __init__(self, crop_size=(128, 128, 128)):
        self.crop_size = crop_size

    def __call__(self, sample):
        image = sample['image3d']

        # 获取形状并考虑通道维度
        D, H, W, C = image.shape

        # 随机裁剪
        crop_z = random.randint(0, D - self.crop_size[0])
        crop_y = random.randint(0, H - self.crop_size[1])
        crop_x = random.randint(0, W - self.crop_size[2])

        cropped_image = image[crop_z:crop_z + self.crop_size[0],
                        crop_y:crop_y + self.crop_size[1],
                        crop_x:crop_x + self.crop_size[2]]

        # 填充，保持尺寸
        pad_z = self.crop_size[0] - cropped_image.shape[0]
        pad_y = self.crop_size[1] - cropped_image.shape[1]
        pad_x = self.crop_size[2] - cropped_image.shape[2]

        cropped_image = np.pad(cropped_image,
                               ((0, pad_z), (0, pad_y), (0, pad_x), (0, 0)),
                               mode='constant')

        sample['image3d'] = cropped_image
        return sample


import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

class RandomElasticDeformation3D(object):
    """
    对 3D 医学影像做弹性形变，保持 shape 不变。
    - 支持 channels_last: (D,H,W,C) 以及 channels_first: (C,D,H,W)
    - 对所有通道应用同一几何形变（保证多模态对齐）
    - 若 sample 中有 'label3d'，会用最近邻 order=0 同步变形
    """
    def __init__(self,
                 alpha=8.0,           # 形变强度（体素单位），典型 5~15
                 sigma=6.0,           # 平滑强度（高斯核标准差，体素），典型 3~8
                 p=0.5,               # 触发概率
                 order=1,             # 图像插值阶次（1=线性），标签会强制用0
                 mode='nearest',      # 边界策略：'nearest'/'reflect'/'constant'等
                 cval=0.0,            # mode='constant' 时填充值
                 channels_last=True   # 你的数据是 (D,H,W,C)，所以默认 True
                 ):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.order = order
        self.mode = mode
        self.cval = cval
        self.channels_last = channels_last

    def __call__(self, sample):
        # 期望 sample 里至少有 'image3d'
        img = sample['image3d']
        if np.random.rand() > self.p:
            return sample  # 不触发增强

        # 统一到 (D,H,W,C) 处理
        transposed = False
        if self.channels_last:
            assert img.ndim in (3,4), "image3d 应为 (D,H,W) 或 (D,H,W,C)"
            if img.ndim == 3:
                img = img[..., None]  # 加通道维
        else:
            # (C,D,H,W) → (D,H,W,C)
            assert img.ndim in (3,4), "image3d 应为 (D,H,W) 或 (C,D,H,W)"
            if img.ndim == 3:
                img = img[None, ...]      # (C,D,H,W) 里的 C=1
            img = np.moveaxis(img, 0, -1) # -> (D,H,W,C)
            transposed = True

        D, H, W, C = img.shape

        # --- 生成 3 个随机位移场（dx, dy, dz），每个形状 (D,H,W) ---
        # 均值为 0、范围约 [-1,1] 的随机场 → 高斯平滑 → 乘 alpha 缩放
        def _rand_field():
            field = np.random.uniform(-1, 1, size=(D, H, W))
            field = gaussian_filter(field, self.sigma, mode='nearest')
            field *= self.alpha
            return field

        dz = _rand_field()
        dy = _rand_field()
        dx = _rand_field()

        # --- 网格坐标 ---
        z, y, x = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W),
            indexing='ij'
        )

        # --- 位移后的坐标（注意顺序：map_coordinates 需要 [z', y', x']）---
        z_new = z + dz
        y_new = y + dy
        x_new = x + dx
        coords = [z_new, y_new, x_new]  # list of (D,H,W)

        # --- 逐通道插值 ---
        img_dtype = img.dtype
        out = np.empty_like(img, dtype=np.float32)

        for c in range(C):
            out[..., c] = map_coordinates(
                img[..., c],
                coordinates=coords,           # 形状 (3, D, H, W) 亦可；list 也可以
                order=self.order,
                mode=self.mode,
                cval=self.cval
            )

        # 恢复 dtype
        out = out.astype(img_dtype, copy=False)

        # --- 同步变形标签（如有），最近邻插值以避免类别混叠 ---
        if 'label3d' in sample and sample['label3d'] is not None:
            lab = sample['label3d']
            lab_transposed = False
            if self.channels_last:
                # 允许标签为 (D,H,W) 或 (D,H,W,1)；多类 one-hot 可为 (D,H,W,K)
                if lab.ndim == 3:
                    lab_c = 1
                    lab_ = lab[..., None]
                elif lab.ndim == 4:
                    lab_c = lab.shape[-1]
                    lab_ = lab
                else:
                    raise ValueError("label3d 形状需为 (D,H,W) 或 (D,H,W,C).")
            else:
                # 允许标签为 (D,H,W) 或 (C,D,H,W)
                if lab.ndim == 3:
                    lab_c = 1
                    lab_ = lab[None, ...]
                elif lab.ndim == 4:
                    lab_c = lab.shape[0]
                    lab_ = lab
                else:
                    raise ValueError("label3d 形状需为 (D,H,W) 或 (C,D,H,W).")
                lab_ = np.moveaxis(lab_, 0, -1)  # -> (D,H,W,C)
                lab_transposed = True

            lab_out = np.empty_like(lab_, dtype=lab_.dtype)
            for c in range(lab_c):
                lab_out[..., c] = map_coordinates(
                    lab_[..., c],
                    coordinates=coords,
                    order=0,                  # 最近邻
                    mode=self.mode,
                    cval=0
                )

            # 去掉单通道的多余维度 & 还原布局
            if lab_c == 1:
                lab_out = lab_out[..., 0]
            if not self.channels_last and lab_transposed:
                if lab_c == 1:
                    lab_out = lab_out  # (D,H,W)
                else:
                    lab_out = np.moveaxis(lab_out, -1, 0)  # (C,D,H,W)
            sample['label3d'] = lab_out

        # 还原图像布局
        if transposed:
            # (D,H,W,C) -> (C,D,H,W)
            out = np.moveaxis(out, -1, 0)
        else:
            if out.shape[-1] == 1:
                out = out[..., 0]  # 如果原来是 (D,H,W) 单通道，移除通道维


        return sample

from skimage import exposure
class RandomBrightnessContrast(object):
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, sample):
        image = sample['image3d']
        # 随机调整亮度
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = image * brightness_factor

        # 随机调整对比度
        p2, p98 = np.percentile(image, (5, 95))
        image = exposure.rescale_intensity(image, in_range=(p2, p98))

        sample['image3d'] = image
        return sample

def transform3d(sample):
    trans = transforms.Compose([
        guiyihua(),
        Pad(),
        RandomCropAndPad(),
        RandomNoiseAugmentor(),
        RandomIntensityChange(),
        Random_rotate(),  # time-consuming
        Random_Flip(),
        ToTensor()
    ])
    return trans(sample)


def transform_valid3d(sample):
    trans = transforms.Compose([
        guiyihua(),
        Pad(),
        ToTensor()
    ])

    return trans(sample)


class process(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                # 只保留路径部分，不重复添加文件名
                path = os.path.join(root, line)  # 注意，这里不要再加上 name，line 已经包含了文件夹结构
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]

        if self.mode == 'train':

            image3d, radiomic, idh, grade = pkload(path + 'idhgrade.pkl')  # 加载文件
            sample = {'image3d': image3d, 'radiomic': radiomic, 'idh': idh, 'grade': grade}
            sample = transform3d(sample)
            sample['radiomic'] = np.array(sample['radiomic']).astype(np.float32)
            return sample['image3d'], sample['radiomic'], sample['idh'], sample['grade']

        elif self.mode == 'valid':
            image3d, radiomic, idh, grade = pkload(path + 'idhgrade.pkl')
            name = os.path.basename(path)
            sample = {'image3d': image3d, 'radiomic': radiomic, 'idh': idh, 'grade': grade}
            sample = transform_valid3d(sample)
            sample['radiomic'] = np.array(sample['radiomic']).astype(np.float32)
            return sample['image3d'], sample['radiomic'], sample['idh'], sample['grade'],name
        else:
            image = pkload(path + 'idhgrade.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]




