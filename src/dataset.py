from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch


## This is a dataset referring to the generative problem
## The Dataset class for FUNSD Dataset (and I believe the same would be use for CORD Dataset)
class FUNSDDs(Dataset):

  def __init__(self, ds, tokenizer, max_seq_length:int = 512, pad_token_box = [0, 0, 0, 0], resize_scale = (512, 384), transform = None):
        
    """
    Args:
      ds (list): list of dict, each dict contains the following keys:
        - image (np.ndarray): the image
        - tokens (list): list of tokens
        - bboxes (list): list of bboxes
        - ner_tags (list): list of ner_tags
      tokenizer (Tokenizer): the tokenizer
      max_seq_length (int, optional): the maximum length of the sequence. Defaults to 512.
      pad_token_box (list, optional): the padding token box. Defaults to [0, 0, 0, 0].
      resize_scale (tuple, optional): the resize scale. Defaults to (512, 384).
      transform (callable, optional): the transform. Defaults to None.
    """

    self.ds = ds
    self.tokenizer = tokenizer
    self.max_seq_length = max_seq_length
    self.pad_token_box = pad_token_box
    self.resize_scale = resize_scale
    self.transform = transform if transform is not None else ToTensor()

  def __len__(self):
    """
    Returns:
      int: the length of the dataset
    """
    return len(self.ds)
  
  def __getitem__(self, idx):
    
    """
    Args:
      idx (int): the index of the data to be returned.
    """

    encoding = self.ds[idx]

    resized_image = encoding['image'].copy().resize(self.resize_scale)
    words = encoding['tokens']
    bboxes = encoding['bboxes']
    labels = encoding['ner_tags']

    ## 1. Performing the image pre-processing
    img_tensor = self.transform(resized_image)  ## (3, 384, 512)

    ## 2. Performing the semantic pre-processing
    encoding = self.tokenizer(words, is_split_into_words = True, add_special_tokens = False)

    # pad_token_box = [0, 0, 0, 0]
    max_seq_length = 512

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    ## Note that, there is no need for bboxes, since the model does not use bbox as feature, so no pre-processing of that
    bbox_according_to_tokenizer = [bboxes[i] for i in encoding.word_ids()]
    # labels_according_to_tokenizer = [self.tokenizer(str(labels[i] + 1))['input_ids'][0] for i in encoding.word_ids()]
    #labels_according_to_tokenizer = [self.tokenizer(str(labels[i] + 1))['input_ids'][0] for i, _ in enumerate(labels)]

    # Truncation of token_boxes + token_labels
    special_tokens_count = 1
    if len(input_ids) > max_seq_length - special_tokens_count:
        bbox_according_to_tokenizer = bbox_according_to_tokenizer[: (max_seq_length - special_tokens_count)]
        input_ids = input_ids[: (max_seq_length - special_tokens_count)]
        #labels_according_to_tokenizer = labels_according_to_tokenizer[: (max_seq_length - special_tokens_count)]
        attention_mask = attention_mask[: (max_seq_length - special_tokens_count)]


    ## Padding
    input_ids =  input_ids + [self.tokenizer.eos_token_id]
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [[1000, 1000, 1000, 1000]]
    #labels_according_to_tokenizer = labels_according_to_tokenizer + [self.tokenizer.eos_token_id] ## For QA, the model requires an end of sentence i.e eos token
    attention_mask = attention_mask + [1]

    pad_length = max_seq_length -  len(input_ids)

    input_ids = input_ids + [self.tokenizer.pad_token_id] * (pad_length)
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [self.pad_token_box] * (pad_length)
    #labels_according_to_tokenizer = labels_according_to_tokenizer + [self.tokenizer.pad_token_id] * (pad_length)
    attention_mask = attention_mask + [0] * (pad_length)

    ## Converting stuffs to tensor
    input_ids = torch.tensor(input_ids)
    bbox_according_to_tokenizer = torch.tensor(bbox_according_to_tokenizer)
    #labels_according_to_tokenizer = torch.tensor(labels_according_to_tokenizer)
    attention_mask = torch.tensor(attention_mask)

    return {"input_ids" : input_ids,  "labels" : labels, "attention_mask" : attention_mask, "bboxes" : bbox_according_to_tokenizer,  # labels_according_to_tokenizer
            "pixel_values" : img_tensor}


## This is a dataset referring to the extractive problem, I believe the same would be used for CORD Dataset, and this was also a mistake, since I didn't  read the last part of the paper properly
class ExtFUNSDDs(Dataset):
  def __init__(self, ds, tokenizer, max_seq_length:int = 512, pad_token_box = [0, 0, 0, 0], resize_scale = (512, 384), transform = None):
        
    """
    Args:
      ds (list): list of dict, each dict contains the following keys:
        - image (np.ndarray): the image
        - tokens (list): list of tokens
        - bboxes (list): list of bboxes
        - ner_tags (list): list of ner_tags
      tokenizer (Tokenizer): the tokenizer
      max_seq_length (int, optional): the maximum length of the sequence. Defaults to 512.
      pad_token_box (list, optional): the padding token box. Defaults to [0, 0, 0, 0].
      resize_scale (tuple, optional): the resize scale. Defaults to (512, 384).
      transform (callable, optional): the transform. Defaults to None.
    """

    self.ds = ds
    self.tokenizer = tokenizer
    self.max_seq_length = max_seq_length
    self.pad_token_box = pad_token_box
    self.resize_scale = resize_scale
    self.transform = transform if transform is not None else ToTensor()

  def __len__(self):
    """
    Returns:
      int: the length of the dataset
    """
    return len(self.ds)
  
  def __getitem__(self, idx):
    
    """
    Args:
      idx (int): the index of the data to be returned.
    """

    encoding = self.ds[idx]

    resized_image = encoding['image'].copy().resize(self.resize_scale)
    words = encoding['tokens']
    bboxes = encoding['bboxes']
    labels = encoding['ner_tags']

    ## 1. Performing the image pre-processing
    img_tensor = self.transform(resized_image)  ## (3, 384, 512)

    ## 2. Performing the semantic pre-processing
    encoding = self.tokenizer(words, is_split_into_words = True, add_special_tokens = False)

    # pad_token_box = [0, 0, 0, 0]
    max_seq_length = 512

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    ## Note that, there is no need for bboxes, since the model does not use bbox as feature, so no pre-processing of that
    bbox_according_to_tokenizer = [bboxes[i] for i in encoding.word_ids()]
    labels_according_to_tokenizer = [labels[i] for i in encoding.word_ids()]  ## Labels have to be in the numerical format
    #labels_according_to_tokenizer = [self.tokenizer(str(labels[i] + 1))['input_ids'][0] for i, _ in enumerate(labels)]

    # Truncation of token_boxes + token_labels
    special_tokens_count = 1
    if len(input_ids) > max_seq_length - special_tokens_count:
        bbox_according_to_tokenizer = bbox_according_to_tokenizer[: (max_seq_length - special_tokens_count)]
        input_ids = input_ids[: (max_seq_length - special_tokens_count)]
        labels_according_to_tokenizer = labels_according_to_tokenizer[: (max_seq_length - special_tokens_count)]
        attention_mask = attention_mask[: (max_seq_length - special_tokens_count)]


    ## Padding
    input_ids =  input_ids + [self.tokenizer.eos_token_id]
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [[1000, 1000, 1000, 1000]]
    labels_according_to_tokenizer = labels_according_to_tokenizer + [-100] ## For QA, the model requires an end of sentence i.e eos token
    attention_mask = attention_mask + [1]

    pad_length = max_seq_length -  len(input_ids)

    input_ids = input_ids + [self.tokenizer.pad_token_id] * (pad_length)
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [self.pad_token_box] * (pad_length)
    labels_according_to_tokenizer = labels_according_to_tokenizer + [-100] * (pad_length)
    attention_mask = attention_mask + [0] * (pad_length)

    ## Converting stuffs to tensor
    input_ids = torch.tensor(input_ids)
    bbox_according_to_tokenizer = torch.tensor(bbox_according_to_tokenizer)
    #labels_according_to_tokenizer = torch.tensor(labels_according_to_tokenizer)
    attention_mask = torch.tensor(attention_mask)

    return {"input_ids" : input_ids,  "labels" : labels, "attention_mask" : attention_mask, "bboxes" : bbox_according_to_tokenizer,  # labels_according_to_tokenizer
            "pixel_values" : img_tensor}