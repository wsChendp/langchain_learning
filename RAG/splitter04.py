import os

from langchain_community.embeddings import ModelScopeEmbeddings, BaichuanTextEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

with open('test.txt', encoding='utf-8') as f:
    text_data = f.read()


os.environ['BAICHUAN_API_KEY'] = 'sk-c3cb1ff363cd14e257a673bc3d9e0d16'
# embeddings = BaichuanTextEmbeddings()
embeddings = ModelScopeEmbeddings(model_id="iic/nlp_gte_sentence-embedding_chinese-base")
text_splitter = SemanticChunker(embeddings,breakpoint_threshold_type='percentile')

docs_list = text_splitter.create_documents([text_data])

print(docs_list[1].page_content)

print(len(docs_list))
