import pandas as pd
import os
from word2vec import word2vecModel, write_CODE_ATC, display_pca_scatterplot
from gensim.models import Word2Vec

MODEL_PATH = "G:\Data\prevention_iatrogenie_centrale_2\Datos_github\Fil Rouge\Fil Rouge\word2vec.model"
PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\securimed_filtered.csv"

if not os.path.isfile(MODEL_PATH):
    print("Creation du modele w2c")
    PATH_ECHANTILLON = "G:\Data\prevention_iatrogenie_centrale_2\echantillon.csv"
    df = pd.read_csv(PATH_ECHANTILLON, delimiter=";")
    write_CODE_ATC(df)
    word2vecModel("CODE_ATC.txt")


model = Word2Vec.load(MODEL_PATH)

# Print tests
#print(model.wv.index_to_key)
context = model.wv.most_similar("n05ba06")

print("Most similar with: 'N05BA06' Skip Gram : ", context)
print([atc[0] for atc in context])
#display_pca_scatterplot(model,sample=10)

#display_pca_scatterplot(model,sample=10)
