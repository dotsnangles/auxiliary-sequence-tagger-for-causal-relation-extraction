import os
import argparse
import pandas as pd

from module.model import tokenizer, id2label
from module.utils import infer, extract

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

""" Cleansing """

# def preprocess(df):
#     df['Event'] = df['Event'].str.replace("．", ".", regex=False)
#     df['Event'] = df['Event'].astype(str)
#     # df['Event'] = df['Event'].str.replace(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]+', " ", regex=True)
#     return df

# data = preprocess(data)

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
toLabel = lambda x: id2label[x]
for input_ids, result in zip(input_ids_lst, infer_results):
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    subwords = tokenizer.encode(decoded)[1:-1]
    subwords = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), subwords))
    subwordsList.append(subwords)

    lenSubwords = len(subwords)    
    result = result[1:lenSubwords+1]
    tags = []
    for id in result:
        tags.append(toLabel(id))
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

df['Date'], df['Time'], df['Location'], df['Work'] = df['Date']+'\t', '\t'+df['Time']+'\t', '\t'+df['Location']+'\t', '\t'+df['Work']
df.to_csv('./inference_result.txt', index=False, sep='|')
print('완료되었습니다.')