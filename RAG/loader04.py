from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(file_path='test_translated.md', encoding='utf-8', mode = 'elements')

data = loader.load_and_split()
print(data)
