from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='test.pdf', extract_images=True)

data = loader.load()
print(data)
