from charset_normalizer.md import is_separator
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('test.txt', encoding='utf-8') as f:
    text_data = f.read()

separators = [
    "\n\n",
    "\n",
    "。", "！", "？", ".", "!", "?",
    ",", "，", ";", "；",
    "、", " ",
    ""
]

# 递归切割器,然乎存到向量数据库中
test_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    separators=separators
)

chunk_list = test_splitter.create_documents([text_data])

print(len(chunk_list))

for chunk in chunk_list[:-1]:
    print(chunk)
    print('---')
