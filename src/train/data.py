import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from io import BytesIO
import glob
from tqdm import tqdm

class EditDataset_with_Omini(Dataset):
    def __init__(
        self,
        magic_dataset,
        omni_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        crop_the_noise: bool = True,
    ):
        self.dataset = [magic_dataset['train'], magic_dataset['dev'], omni_dataset]

        from collections import Counter
        tasks = omni_dataset['task']
        task_counts = Counter(tasks)
        print("\n task type statistic：")
        for task, count in task_counts.items():
            print(f"{task}: {count} data ({count/len(tasks)*100:.2f}%)")
            
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.crop_the_noise = crop_the_noise
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset[0]) + len(self.dataset[1]) + len(self.dataset[2])


    def __getitem__(self, idx):
        split = 0 if idx < len(self.dataset[0]) else (1 if idx < len(self.dataset[0]) + len(self.dataset[1]) else 2)
        
        if idx >= len(self.dataset[0]) + len(self.dataset[1]):
            idx -= len(self.dataset[0]) + len(self.dataset[1])
        elif idx >= len(self.dataset[0]):
            idx -= len(self.dataset[0])
            
        image = self.dataset[split][idx]["source_img" if split != 2 else "src_img"]
        instruction = 'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but ' + (self.dataset[split][idx]["instruction"] if split != 2 else random.choice(self.dataset[split][idx]["edited_prompt_list"]))
        edited_image = self.dataset[split][idx]["target_img" if split != 2 else "edited_img"]
     
        if self.crop_the_noise and split <= 1:
            image = image.crop((0, 0, image.width, image.height - image.height // 32))
            edited_image = edited_image.crop((0, 0, edited_image.width, edited_image.height - edited_image.height // 32))
        
        image = image.resize((self.condition_size, self.condition_size)).convert("RGB")
        edited_image = edited_image.resize((self.target_size, self.target_size)).convert("RGB")

        combined_image = Image.new('RGB', (self.condition_size * 2, self.condition_size))
        combined_image.paste(image, (0, 0))
        combined_image.paste(edited_image, (self.condition_size, 0))
        
       
            
        mask = Image.new('L', (self.condition_size * 2, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        mask_combined_image = combined_image.copy()
        draw = ImageDraw.Draw(mask_combined_image)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        if random.random() < self.drop_text_prob:
            instruction = " "

        return {
            "image": self.to_tensor(combined_image), 
            "condition": self.to_tensor(mask),
            "condition_type": "edit",  
            "description": instruction,
            "position_delta": np.array([0, 0]),
            **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
        }

class OminiDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        specific_task: list = None,
    ):
        self.base_dataset = base_dataset['train']
        if specific_task is not None:
            self.specific_task = specific_task
            task_indices = [i for i, task in enumerate(self.base_dataset['task']) if task in self.specific_task]
            task_set = set([task for task in self.base_dataset['task']])
            ori_len = len(self.base_dataset)
            self.base_dataset = self.base_dataset.select(task_indices)
            print(specific_task, len(self.base_dataset), ori_len)
            print(task_set)
            
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()        

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["src_img"]
        instruction = 'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but ' + random.choice(self.base_dataset[idx]["edited_prompt_list"])
            
        edited_image = self.base_dataset[idx]["edited_img"]
        
        image = image.resize((self.condition_size, self.condition_size)).convert("RGB")
        edited_image = edited_image.resize((self.target_size, self.target_size)).convert("RGB")

        combined_image = Image.new('RGB', (self.condition_size * 2, self.condition_size))
        combined_image.paste(image, (0, 0))
        combined_image.paste(edited_image, (self.condition_size, 0))
        
        mask = Image.new('L', (self.condition_size * 2, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        mask_combined_image = combined_image.copy()
        draw = ImageDraw.Draw(mask_combined_image)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        if random.random() < self.drop_text_prob:
            instruction = ""

        return {
            "image": self.to_tensor(combined_image),
            "condition": self.to_tensor(mask),
            "condition_type": "edit", 
            "description": instruction,
            "position_delta": np.array([0, 0]),
            **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
        }


class EditDataset_mask(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        crop_the_noise: bool = True,
    ):
        print('THIS IS MAGICBRUSH!')
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.crop_the_noise = crop_the_noise
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset['train']) + len(self.base_dataset['dev'])

    def rgba_to_01_mask(image_rgba: Image.Image, reverse: bool = False, return_type: str = "numpy"):
        """
        Convert an RGBA image to a binary mask with values in the range [0, 1], where 0 represents transparent areas
        and 1 represents non-transparent areas. The resulting mask has a shape of (1, H, W).

        :param image_rgba: An RGBA image in PIL format.
        :param reverse: If True, reverse the mask, making transparent areas 1 and non-transparent areas 0.
        :param return_type: Specifies the return type. "numpy" returns a NumPy array, "PIL" returns a PIL Image.

        :return: The binary mask as a NumPy array or a PIL Image in RGB format.
        """
        alpha_channel = np.array(image_rgba)[:, :, 3]
        image_bw = (alpha_channel != 255).astype(np.uint8)
        if reverse:
            image_bw = 1 - image_bw
        mask = image_bw
        if return_type == "numpy":
            return mask
        else: # return PIL image
            mask = Image.fromarray(np.uint8(mask * 255) , 'L').convert('RGB')
            return mask

    def __getitem__(self, idx):
        split = 'train' if idx < len(self.base_dataset['train']) else 'dev'
        idx = idx - len(self.base_dataset['train']) if split == 'dev' else idx
        image = self.base_dataset[split][idx]["source_img"]
        instruction = 'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left. \n ' + self.base_dataset[split][idx]["instruction"]
        edited_image = self.base_dataset[split][idx]["target_img"]
        
        if self.crop_the_noise:
            image = image.crop((0, 0, image.width, image.height - image.height // 32))
            edited_image = edited_image.crop((0, 0, edited_image.width, edited_image.height - edited_image.height // 32))
        
        image = image.resize((self.condition_size, self.condition_size)).convert("RGB")
        edited_image = edited_image.resize((self.target_size, self.target_size)).convert("RGB")

        combined_image = Image.new('RGB', (self.condition_size * 2, self.condition_size))
        combined_image.paste(image, (0, 0))
        combined_image.paste(edited_image, (self.condition_size, 0))
        
        mask = Image.new('L', (self.condition_size * 2, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        mask_combined_image = combined_image.copy()
        draw = ImageDraw.Draw(mask_combined_image)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        if random.random() < self.drop_text_prob:
            instruction = " \n "
        return {
            "image": self.to_tensor(combined_image), 
            "condition": self.to_tensor(mask),
            "condition_type": "edit_n",
            "description": instruction,
            "position_delta": np.array([0, 0]),
            **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
        }
        
class EditDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        crop_the_noise: bool = True,
    ):
        print('THIS IS MAGICBRUSH!')
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.crop_the_noise = crop_the_noise
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset['train']) + len(self.base_dataset['dev'])

    def rgba_to_01_mask(image_rgba: Image.Image, reverse: bool = False, return_type: str = "numpy"):
        """
        Convert an RGBA image to a binary mask with values in the range [0, 1], where 0 represents transparent areas
        and 1 represents non-transparent areas. The resulting mask has a shape of (1, H, W).

        :param image_rgba: An RGBA image in PIL format.
        :param reverse: If True, reverse the mask, making transparent areas 1 and non-transparent areas 0.
        :param return_type: Specifies the return type. "numpy" returns a NumPy array, "PIL" returns a PIL Image.

        :return: The binary mask as a NumPy array or a PIL Image in RGB format.
        """
        alpha_channel = np.array(image_rgba)[:, :, 3]
        image_bw = (alpha_channel != 255).astype(np.uint8)
        if reverse:
            image_bw = 1 - image_bw
        mask = image_bw
        if return_type == "numpy":
            return mask
        else: # return PIL image
            mask = Image.fromarray(np.uint8(mask * 255) , 'L').convert('RGB')
            return mask

    def __getitem__(self, idx):
        split = 'train' if idx < len(self.base_dataset['train']) else 'dev'
        idx = idx - len(self.base_dataset['train']) if split == 'dev' else idx
        image = self.base_dataset[split][idx]["source_img"]
        instruction = 'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but ' + self.base_dataset[split][idx]["instruction"]
        edited_image = self.base_dataset[split][idx]["target_img"]
        
       
        if self.crop_the_noise:
            image = image.crop((0, 0, image.width, image.height - image.height // 32))
            edited_image = edited_image.crop((0, 0, edited_image.width, edited_image.height - edited_image.height // 32))
        
        image = image.resize((self.condition_size, self.condition_size)).convert("RGB")
        edited_image = edited_image.resize((self.target_size, self.target_size)).convert("RGB")

        combined_image = Image.new('RGB', (self.condition_size * 2, self.condition_size))
        combined_image.paste(image, (0, 0))
        combined_image.paste(edited_image, (self.condition_size, 0))
        
        mask = Image.new('L', (self.condition_size * 2, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        mask_combined_image = combined_image.copy()
        draw = ImageDraw.Draw(mask_combined_image)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)
        
        if random.random() < self.drop_text_prob:
            instruction = " "

        return {
            "image": self.to_tensor(combined_image), 
            "condition": self.to_tensor(mask),
            "condition_type": "edit",
            "description": instruction, 
            "position_delta": np.array([0, 0]),
            **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
        }


class LogoDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str,
        split: str = 'train',
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.dataset_folder = dataset_folder
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

        mask = Image.new('L', (self.condition_size * 2, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([self.condition_size, 0, self.condition_size * 2, self.condition_size], fill=255)

        image_paths = glob.glob(os.path.join(dataset_folder, split, "*.png"))
        image_paths.sort()

        self.data = []
        for fn in tqdm(image_paths, desc=f"Loading {split} images"):
            combined_image = Image.open(fn).convert("RGB").resize((self.condition_size * 2, self.condition_size))
            image = combined_image.crop((0, 0, combined_image.width // 2, combined_image.height))
            edited_image = combined_image.crop((combined_image.width // 2, 0, combined_image.width, combined_image.height))

            with open(fn.replace('.png', '.txt'), 'r') as f:
                instruction = f.read().strip()

            self.data.append(
                {
                "image": self.to_tensor(combined_image), 
                "condition": self.to_tensor(mask),
                "condition_type": "logo_decomposition",
                "description": instruction, 
                "position_delta": np.array([0, 0]),
                **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range for LogoDataset")
        
        data = self.data[idx]
        
        if random.random() < self.drop_text_prob:
            data["description"] = ""
        
        return data
    

class LogoProductDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str,
        condition_type: str,
        split: str = 'train',
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.dataset_folder = dataset_folder
        self.condition_type = condition_type
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

        mask = Image.new('L', (self.condition_size * 3, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)

        if self.condition_type.endswith("decomposition"):
            draw.rectangle([self.condition_size, 0, self.condition_size * 3, self.condition_size], fill=255)
        elif self.condition_type.endswith("composition"):
            draw.rectangle([self.condition_size * 2, 0, self.condition_size * 3, self.condition_size], fill=255)
        else:
            raise ValueError("Unknown condition_type")

        image_paths = glob.glob(os.path.join(dataset_folder, split, "*.png"))
        image_paths.sort()

        self.data = []
        for fn in tqdm(image_paths, desc=f"Loading {split} images"):
            combined_image = Image.open(fn).convert("RGB").resize((self.condition_size * 3, self.condition_size))
            image = combined_image.crop((0, 0, combined_image.width // 3, combined_image.height))
            edited_image = combined_image.crop((combined_image.width // 3, 0, combined_image.width, combined_image.height))

            with open(fn.replace('.png', '.txt'), 'r') as f:
                instruction = f.read().strip()

            self.data.append(
                {
                "image": self.to_tensor(combined_image), 
                "condition": self.to_tensor(mask),
                "condition_type": self.condition_type,
                "description": instruction, 
                "position_delta": np.array([0, 0]),
                **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range for LogoProductDataset")

        data = self.data[idx]
        
        if random.random() < self.drop_text_prob:
            data["description"] = ""
        
        return data
    

class FGBGDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str,
        condition_type: str,
        split: str = 'train',
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.dataset_folder = dataset_folder
        self.condition_type = condition_type
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

        mask = Image.new('L', (self.condition_size * 3, self.condition_size), 0)
        draw = ImageDraw.Draw(mask)

        if self.condition_type.endswith("decomposition"):
            draw.rectangle([self.condition_size, 0, self.condition_size * 3, self.condition_size], fill=255)
        elif self.condition_type.endswith("composition"):
            draw.rectangle([self.condition_size * 2, 0, self.condition_size * 3, self.condition_size], fill=255)
        else:
            raise ValueError("Unknown condition_type")

        image_paths = glob.glob(os.path.join(dataset_folder, split, "*.png"))
        image_paths.sort()

        self.data = []
        for fn in tqdm(image_paths, desc=f"Loading {split} images"):
            combined_image = Image.open(fn).convert("RGB").resize((self.condition_size * 3, self.condition_size))
            image = combined_image.crop((0, 0, combined_image.width // 3, combined_image.height))
            edited_image = combined_image.crop((combined_image.width // 3, 0, combined_image.width, combined_image.height))

            with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                instruction = f.read().strip()

            self.data.append(
                {
                "image": self.to_tensor(combined_image), 
                "condition": self.to_tensor(mask),
                "condition_type": self.condition_type,
                "description": instruction, 
                "position_delta": np.array([0, 0]),
                **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range for FGBGDataset")

        data = self.data[idx]
        
        if random.random() < self.drop_text_prob:
            data["description"] = ""
        
        return data


class CycleDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str,
        condition_type: str,
        split: str = 'train',
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.dataset_folder = dataset_folder
        self.condition_type = condition_type
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

        assert "cycle" in self.condition_type

        mask_decomp = Image.new('L', (self.condition_size * 3, self.condition_size), 0)
        draw_decomp = ImageDraw.Draw(mask_decomp)
        draw_decomp.rectangle([self.condition_size, 0, self.condition_size * 3, self.condition_size], fill=255)

        mask_comp = Image.new('L', (self.condition_size * 3, self.condition_size), 0)
        draw_comp = ImageDraw.Draw(mask_comp)
        draw_comp.rectangle([0, 0, self.condition_size, self.condition_size], fill=255)

        image_paths = glob.glob(os.path.join(dataset_folder, split, "*.png"))
        image_paths.sort()
        self.image_paths = image_paths

        self.data = []
        for fn in tqdm(image_paths, desc=f"Loading {split} images"):
            combined_image = Image.open(fn).convert("RGB").resize((self.condition_size * 3, self.condition_size))
            # image = combined_image.crop((0, 0, combined_image.width // 3, combined_image.height))
            edited_image = combined_image.crop((combined_image.width // 3, 0, combined_image.width, combined_image.height))

            with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                instruction_decomp = f.read().strip()
            with open(fn.replace('.png', '_comp.txt'), 'r') as f:
                instruction_comp = f.read().strip()
            self.data.append(
                {
                    "decomp": {
                        "image": self.to_tensor(combined_image), 
                        "condition": self.to_tensor(mask_decomp),
                        "condition_type": "decomposition",
                        "description": instruction_decomp, 
                        "position_delta": np.array([0, 0]),
                        **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
                    },
                    "comp": {
                        "image": self.to_tensor(combined_image), 
                        "condition": self.to_tensor(mask_comp),
                        "condition_type": "composition",
                        "description": instruction_comp, 
                        "position_delta": np.array([0, 0]),
                        **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
                    }
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range for LogoProductDataset")

        data = self.data[idx]
        
        if random.random() < self.drop_text_prob:
            data["description"] = ""
        
        return data


# import os
# import glob
# import random
# import numpy as np
# from tqdm import tqdm
# from PIL import Image, ImageDraw
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as T

# class CycleDataset(Dataset):
#     def __init__(
#         self,
#         dataset_folder: str,
#         condition_type: str,
#         split: str = 'train',
#         condition_size: int = 512,
#         target_size: int = 512,
#         drop_text_prob: float = 0.1,
#         return_pil_image: bool = False,
#     ):
#         self.dataset_folder = dataset_folder
#         self.condition_type = condition_type
#         self.condition_size = condition_size
#         self.target_size = target_size
#         self.drop_text_prob = drop_text_prob
#         self.return_pil_image = return_pil_image
#         self.to_tensor = T.ToTensor()

#         assert "cycle" in self.condition_type, "condition_type must contain 'cycle'"

#         # 预先生成掩码（只用一份，避免占内存）
#         self.mask_decomp = self._create_mask(decomp=True)
#         self.mask_comp = self._create_mask(decomp=False)

#         # 收集路径
#         self.samples = []
#         image_paths = sorted(glob.glob(os.path.join(dataset_folder, split, "*.png")))
#         for fn in tqdm(image_paths, desc=f"Indexing {split} images"):
#             decomp_txt = fn.replace('.png', '_decomp.txt')
#             comp_txt = fn.replace('.png', '_comp.txt')
#             self.samples.append({
#                 "image_path": fn,
#                 "decomp_txt": decomp_txt,
#                 "comp_txt": comp_txt
#             })

#     def _create_mask(self, decomp=True):
#         mask = Image.new('L', (self.condition_size * 3, self.condition_size), 0)
#         draw = ImageDraw.Draw(mask)
#         if decomp:
#             draw.rectangle([self.condition_size, 0, self.condition_size * 3, self.condition_size], fill=255)
#         else:
#             draw.rectangle([0, 0, self.condition_size, self.condition_size], fill=255)
#         return self.to_tensor(mask)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         if idx >= len(self.samples):
#             raise IndexError("Index out of range for CycleDataset")

#         sample_info = self.samples[idx]
#         image_path = sample_info["image_path"]
#         decomp_txt_path = sample_info["decomp_txt"]
#         comp_txt_path = sample_info["comp_txt"]

#         # 按需加载图片
#         combined_image = Image.open(image_path).convert("RGB").resize((self.condition_size * 3, self.condition_size))
#         edited_image = combined_image.crop(
#             (combined_image.width // 3, 0, combined_image.width, combined_image.height)
#         )

#         # 加载文本说明
#         with open(decomp_txt_path, 'r') as f:
#             instruction_decomp = f.read().strip()
#         with open(comp_txt_path, 'r') as f:
#             instruction_comp = f.read().strip()

#         # 可能随机丢掉文本
#         if random.random() < self.drop_text_prob:
#             instruction_decomp = ""
#         if random.random() < self.drop_text_prob:
#             instruction_comp = ""

#         data = {
#             "decomp": {
#                 "image": self.to_tensor(combined_image),
#                 "condition": self.mask_decomp,
#                 "condition_type": "decomposition",
#                 "description": instruction_decomp,
#                 "position_delta": np.array([0, 0]),
#             },
#             "comp": {
#                 "image": self.to_tensor(combined_image),
#                 "condition": self.mask_comp,
#                 "condition_type": "composition",
#                 "description": instruction_comp,
#                 "position_delta": np.array([0, 0]),
#             }
#         }

#         if self.return_pil_image:
#             data["decomp"]["pil_image"] = [edited_image, combined_image]
#             data["comp"]["pil_image"] = [edited_image, combined_image]

#         return data
