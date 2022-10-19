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
    # if tag does not exists then notify and pass
    # if not '{tag}_I' then stop
    tag_start = f'{tagtype}_B'
    if tag_start in tags:
        b_idx = tags.index(tag_start)
        for i, tag in enumerate(tags[b_idx:]):
            if i > 0 and tag != f'{tagtype}_I':
                tagged = subwords[b_idx:b_idx+i]
                tagged = tokenizer.convert_tokens_to_string(tagged)
                return tagged
    else:
        msg = f'{tagtype} not found.'
        return msg