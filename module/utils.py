import torch
from .model import model as tagger, tokenizer, device

def infer(input_ids_lst, input_masks_lst):
    tagger.eval()
    with torch.no_grad():
        infer_results = []
        for input_ids, input_masks in zip(input_ids_lst, input_masks_lst): 
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            mask_tensor = torch.LongTensor(input_masks).unsqueeze(0).to(device)
            output = tagger(input_tensor, mask_tensor)
            pred = torch.argmax(output, dim=-1).squeeze().detach().cpu().tolist()
            infer_results.append(pred)
        return infer_results
    
def extract(tagtype, tags, subwords):
    tag_start = f'{tagtype}_B'
    tags_idx_dic = {i:x for i,x in zip(range(len(tags)), tags) if x == tag_start}
    tag_start_ids = list(tags_idx_dic.keys())
    if tag_start_ids:
        taggedList = []
        for b_idx in tag_start_ids:
            for i, tag in enumerate(tags[b_idx:]):
                if i > 0 and tag != f'{tagtype}_I':
                    tagged = subwords[b_idx:b_idx+i]
                    tagged = tokenizer.convert_tokens_to_string(tagged)
                    taggedList.append(tagged)
                    break
        if tagtype == 'DAT' or tagtype == 'TIM':
            taggedList = [x for x in taggedList if not len(x) < 2]
            taggedList = [taggedList[0]]
        return ' [SEP] '.join(taggedList)
    else:
        msg = f'{tagtype} not found.'
        return msg       