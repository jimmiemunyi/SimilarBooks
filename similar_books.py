# imports
import streamlit as st
from fastai.tabular.all import *
from PIL import Image

# Preprocessing of App
path = Path()
learn_inf = load_learner(path/'final_model.pkl', cpu=True)
book_factors = learn_inf.model.i_weight.weight
img = Image.open('header.png')
books = pd.read_csv('books.csv')


def selectbox_with_default(text, values, default, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

def get_similar_books(title, number):
    idx = learn_inf.dls.classes['original_title'].o2i[title]
    distances = nn.CosineSimilarity(dim=1)(book_factors, book_factors[idx][None])
    idx = distances.argsort(descending=True)[1:number+1]
    similar = [learn_inf.dls.classes['original_title'][i] for i in idx]
    ids = [int(books.loc[books['original_title']==str(i)]['goodreads_book_id'].values[0]) for i in similar]
    urls = [f'https://www.goodreads.com/book/show/{id}' for id in ids]
    return similar, urls


# APP
st.image(img, width=200)
st.title('SIMILAR BOOKS')
st.subheader('A Book Recommendation System')
"Here's the [GitHub](https://github.com/jimmiemunyi/SimilarBooks) repo."

st.info("Start typing and you will get suggestions of Books we currently have. We Currently have support for 10, 000 Books!")
title = selectbox_with_default("Which Book Do you want Recommendations From:",
                            books['original_title'], default='Select A Book')
number = st.slider("How many Similar Books do you want?", 1, 10, value=5)

if(st.button("Suggest Similar Books")):
    similar, urls = get_similar_books(title, number)
    st.subheader('Here are your Book Recommendations. Enjoy!')
    for book, url in zip(similar, urls):
        st.write(f'{book}: {url}')

st.title('Developer Details')
'''

My name is Jimmie Munyi. You can connect with me on [Twitter](https://twitter.com/jimmie_munyi). You can check out other projects I have done from [My GitHub](https://github.com/jimmiemunyi) and from [My Blog](https://jimmiemunyi.github.io/blog/).

If you wish to see how Similar Books was created, read this [blog post](https://jimmiemunyi.github.io/blog/projects/tutorial/2021/02/15/Book-Recommendation-Model-Training.html).

'''


