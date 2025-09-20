from langchain_text_splitters import HTMLHeaderTextSplitter

label_split = [
    ('h1', 'Header 1'),
    ('h2', 'Header 2'),
    ('h3', 'Header 3'),
    ('h4', 'Header 4'),
]

with open('test.html', encoding='utf-8') as f:
    html_str = f.read()

html_split = HTMLHeaderTextSplitter(label_split)

docs_list = html_split.split_text_from_url('https://plato.stanford.edu/entries/goedel/')

print('---------------')
print(docs_list[-1])

print('总共切割成:', len(docs_list), '块')
