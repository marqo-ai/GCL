from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import io


class MMEvalDataset(Dataset):
    def __init__(self, image_list, text_list, tokenizer, preprocess):
        self.image_list = image_list
        self.text_list = text_list
        self.tokenize = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        text = self.text_list[idx]
        image = self.preprocess(Image.open(img_path).convert('RGB'))
        text_tokens = self.tokenize(text)

        return image, text_tokens


class MFRightEvalDataset(Dataset):
    def __init__(self, doc_meta_list, tokenizer, preprocess, args, side=0):
        self.img_or_txt = args.img_or_txt

        if side == 0:
            side_keys = args.left_keys
            side_weights = args.left_weights
        else:
            side_keys = args.right_keys
            side_weights = args.right_weights

        self.right_inputs = [[doc[key] for key in side_keys] for doc in doc_meta_list]

        self.right_weights = side_weights
        self.transforms = preprocess

        self.tokenize = tokenizer
        self.context_length = args.context_length
        self.side = side
        self.is_hf_dataset = hasattr(args, 'hf_dataset') and args.hf_dataset

    def __len__(self):
        return len(self.right_inputs)

    def __getitem__(self, idx):
        rights = []
        for j, cat in enumerate(self.img_or_txt[self.side]):
            if cat == "txt":
                rights.append(self.tokenize([str(self.right_inputs[idx][j])], context_length=self.context_length)[0])
            else:
                img = str(self.right_inputs[idx][j]) if not self.is_hf_dataset else io.BytesIO(self.right_inputs[idx][j]['bytes'])
                rights.append(self.transforms(Image.open(img)))

        right_weight = np.array(self.right_weights)

        for i in range(1, len(right_weight)):
            if not pd.notna(self.right_inputs[idx][i]):
                right_weight[i] = 0

        right_weight = right_weight / sum(right_weight)
        return rights, right_weight


# class MFRightEvalDatasetRaw(Dataset):
#     def __init__(self, doc_meta_list, args, side=0):
#         self.img_or_txt = args.img_or_txt
#
#         if side == 0:
#             side_keys = args.left_keys
#             side_weights = args.left_weights
#         else:
#             side_keys = args.right_keys
#             side_weights = args.right_weights
#
#         self.right_inputs = [[doc[key] for key in side_keys] for doc in doc_meta_list]
#
#         self.right_weights = side_weights
#         self.context_length = args.context_length
#         self.side = side
#         self.is_hf_dataset = args.hf_dataset is not None
#
#     def __len__(self):
#         return len(self.right_inputs)
#
#     def __getitem__(self, idx):
#         rights = []
#         for j, cat in enumerate(self.img_or_txt[self.side]):
#             if cat == "txt":
#                 rights.append(str(self.right_inputs[idx][j]))
#             else:
#                 img = str(self.right_inputs[idx][j]) if not self.is_hf_dataset else io.BytesIO(self.right_inputs[idx][j]['bytes'])
#                 rights.append(img)
#
#         right_weight = np.array(self.right_weights)
#
#         for i in range(1, len(right_weight)):
#             if not pd.notna(self.right_inputs[idx][i]):
#                 right_weight[i] = 0
#
#         right_weight = right_weight / sum(right_weight)
#         return rights, right_weight
#
#
# class MFRightEvalDatasetSingleProc(Dataset):
#     def __init__(self, doc_meta_list, args, preprocess, side=0):
#         self.img_or_txt = args.img_or_txt
#
#         if side == 0:
#             side_keys = args.left_keys
#             side_weights = args.left_weights
#         else:
#             side_keys = args.right_keys
#             side_weights = args.right_weights
#
#         self.right_inputs = [[doc[key] for key in side_keys] for doc in doc_meta_list]
#
#         self.right_weights = side_weights
#         self.context_length = args.context_length
#         self.side = side
#         self.is_hf_dataset = args.hf_dataset is not None
#         self.preprocess = preprocess
#
#     def __len__(self):
#         return len(self.right_inputs)
#
#     def __getitem__(self, idx):
#         rights = []
#         for j, cat in enumerate(self.img_or_txt[self.side]):
#             if cat == "txt":
#                 rights.append(self.preprocess(text=str(self.right_inputs[idx][j]), padding='max_length', return_tensors="pt")['input_ids'][0])
#             else:
#                 img = self.preprocess(images=[Image.open(io.BytesIO(self.right_inputs[idx][j]['bytes'])).convert('RGB')], padding='max_length', return_tensors="pt")['pixel_values'][0]
#                 rights.append(img)
#
#         right_weight = np.array(self.right_weights)
#
#         for i in range(1, len(right_weight)):
#             if not pd.notna(self.right_inputs[idx][i]):
#                 right_weight[i] = 0
#
#         right_weight = right_weight / sum(right_weight)
#         return rights, right_weight


class MFRightEvalDatasetTest(Dataset):
    def __init__(self, df, img_or_txt, right_keys, right_weights, tokenizer, preprocess):
        self.img_or_txt = img_or_txt
        self.right_inputs = [df[key].tolist() for key in right_keys]

        self.right_inputs = list(zip(*self.right_inputs))

        self.right_weights = right_weights
        self.transforms = preprocess

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.right_inputs)

    def __getitem__(self, idx):
        rights = []
        for j, cat in enumerate(self.img_or_txt[1]):
            if cat == "txt":
                rights.append(str(self.right_inputs[idx][j]))
            else:
                rights.append(self.transforms(Image.open(str(self.right_inputs[idx][j]))))

        right_weight = np.array(self.right_weights)

        for i in range(1, len(right_weight)):
            if not pd.notna(self.right_inputs[idx][i]):
                right_weight[i] = 0

        right_weight = right_weight / sum(right_weight)
        return rights, right_weight
