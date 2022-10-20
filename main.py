import os
import argparse
import pandas as pd

from module.model import tokenizer, id2label
from module.utils import infer, extract

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", dest="keyword", action="store")
    args = parser.parse_args()

    file_list = os.listdir('./inference_data')
    file_path = []
    for file_name in file_list:
        tmp_path = os.path.join('./inference_data', file_name)
        file_path.append(tmp_path)
    file_path.sort()

    load_df = lambda x: pd.read_csv(x)
    dfs = [load_df(x) for x in file_path]
    data = dfs[0]
    data.dropna(inplace=True)

    keyword = args.keyword
    print(f'\n{keyword}(으)로 검색한 항목들에 대해 개체명 인식을 시작합니다.')

    data['Indexes'] = data.Event.str.find(keyword)
    infer_data = data[data.Indexes > -1]

    infer_data_lst = infer_data.Event.apply(lambda x: tokenizer(x, max_length=256, padding='max_length', truncation=True))

    input_ids_lst = []
    input_masks_lst = []
    for e in infer_data_lst:
        input_ids_lst.append(e['input_ids'])
        input_masks_lst.append(e['attention_mask'])

    infer_results = infer(input_ids_lst, input_masks_lst)

    subwordsList = []
    tagsList = []
    for input_ids, result in zip(input_ids_lst, infer_results):
        subwords = [x for x in input_ids if x != 0][1:-1]
        subwords = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), subwords))
        subwordsList.append(subwords)
        lenSubwords = len(subwords)    

        result = result[1:lenSubwords+1]
        tags = []
        for id in result:
            tags.append(id2label[id])
        tagsList.append(tags)

    tagtypes = dict(
        dat='DAT',
        tim='TIM',
        loc='LOC',
        wrk='WRK',    
    )

    nerResults = []
    count = 0
    for subwords, tags in zip(subwordsList, tagsList):
        extracted = []
        for tagtype in tagtypes.values():
            tagged = extract(tagtype, tags, subwords)
            extracted.append(tagged)
        nerResults.append(extracted)
        count += 1

    df = pd.DataFrame(nerResults, columns=['Date', 'Time', 'Location', 'Work'])
    
    df.to_csv(f'./inference_results/{keyword}_ner.csv', index=False, encoding='utf-8-sig')
    print('완료되었습니다.')
    
if __name__ == '__main__':
    main()