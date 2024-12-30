import os 
import torch
import random 
import pickle
import numpy as np  
import torchvision 
from PIL import Image 
from tqdm import tqdm, trange 
 
# from gluoncv import data as gluoncvData

class COCOSegmentation():
    """COCO Semantic Segmentation Dataset for VOC Pre-training.

    Parameters
    ----------
    root : string
        Path to COCO folder. Default is '$(HOME)/mxnet/datasets/coco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image

    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]
    NUM_CLASS = 21
    CLASSES = ("background", "airplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorcycle", "person", "potted-plant", "sheep", "sofa", "train",
               "tv")

    def __init__(self, root=os.path.expanduser('/home4/solomon/Dataset/IMAGENET/coco_minitrain_25k/images/'),
                 split='train', mode=None, transform=None,ids=None,seed_value=0, **kwargs):
        # super(COCOSegmentation, self).__init__(root, split, mode, transform, **kwargs)
       
        from pycocotools.coco import COCO
        from pycocotools import mask
        self.mode = split
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/instances_minitrain2017.json')
            ids_file = os.path.join(root, 'annotations/train_ids.mx')
            self.root = os.path.join(root, 'train2017')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
            ids_file = os.path.join(root, 'annotations/val_ids.mx')
            self.root = os.path.join(root, 'val2017')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if ids is None:
            if os.path.exists(ids_file):
                with open(ids_file, 'rb') as f:
                    self.ids = pickle.load(f)
            else:
                ids = list(self.coco.imgs.keys())
                self.ids = self._preprocess(ids, ids_file)
        else: self.ids=ids
        self.transform = transform

        # self.random_resize = torchvision.transforms.RandomResizedCrop(480)
        self.center_resize = torchvision.transforms.CenterCrop(480)
        self.seed_value = seed_value
    

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self.__sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self.__sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self.__img_transform(img), self.__mask_transform(mask)
        # general resize, normalize and toTensor
        
   
        return img, mask
    
    def __img_transform(self, img):
        return np.array(img)
    
    def __mask_transform(self,img):
        return np.array(img)
    
    def __sync_transform(self, img, mask):
        if  self.mode == 'train': 
            if self.transform is not None:
                random.seed(self.seed_value)
                torch.manual_seed(self.seed_value)
                img = self.transform(img)
                random.seed(self.seed_value)
                torch.manual_seed(self.seed_value)
                mask = self.transform(mask)
            else:
                img = self.center_resize(img)
                mask = self.center_resize(mask)

        else:    
            img = self.center_resize(img)
            mask = self.center_resize(mask)

        img, mask = self.__img_transform(img), self.__mask_transform(mask)
        return img, mask

    def mask_to_class(self,mask):
        target = torch.from_numpy(mask)
        h,w = target.shape[0],target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        colors = torch.unique(target.view(-1,target.size(2)),dim=0).numpy()
        target = target.permute(2, 0, 1).contiguous()
        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
        for k in mapping:
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3) 
            masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
        return masks.numpy()

    def __len__(self):
        return len(self.ids)

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.\
                format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES
