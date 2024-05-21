import pandas as pd
import numpy as np
import ast

# Configuración de pandas para mostrar todas las columnas en la salida
pd.set_option("display.max_columns", None)

# Cargar los archivos CSV
librosD = pd.read_csv('books_data.csv')  # Carga el archivo 'books_data.csv' en un DataFrame
librosR = pd.read_csv('books_rating.csv')  # Carga el archivo 'books_rating.csv' en otro DataFrame

# Fusionar los DataFrames en la columna "Title"
librosD = librosD.merge(librosR, on='Title')  # Une ambos DataFrames basándose en la columna 'Title'

# Eliminar filas con valores NaN
librosD.dropna(inplace=True)  # Elimina las filas que tienen valores NaN

# Definir las columnas que se desean mantener en el nuevo DataFrame
columnas_necesarias = ['Title', 'description', 'authors', 'publisher', 'categories', 'review/summary', 'review/text']

# Crear una copia del DataFrame con solo las columnas necesarias
libros = librosD[columnas_necesarias].copy()

# Limpiar las columnas de texto eliminando caracteres no deseados
libros['authors'] = libros['authors'].str.replace("[", "").str.replace("]", "").str.replace("'", "")
libros['categories'] = libros['categories'].str.replace("[", "").str.replace("]", "").str.replace("'", "")
libros['description'] = libros['description'].str.replace("[", "").str.replace("]", "").str.replace("'", "")

# Dividir las cadenas de texto en listas de palabras
libros['description'] = libros['description'].apply(lambda x: x.split())
libros['authors'] = libros['authors'].apply(lambda x: x.split())
libros['publisher'] = libros['publisher'].apply(lambda x: x.split())
libros['categories'] = libros['categories'].apply(lambda x: x.split())
libros['review/summary'] = libros['review/summary'].apply(lambda x: x.split())
libros['review/text'] = libros['review/text'].apply(lambda x: x.split())

# Definir una función para eliminar espacios no deseados dentro de las listas de palabras
def dEspacio(x):
    return [i.replace(" ", "") for i in x]

# Aplicar la función de eliminación de espacios a las columnas pertinentes
libros['description'] = libros['description'].apply(dEspacio)
libros['authors'] = libros['authors'].apply(dEspacio)
libros['publisher'] = libros['publisher'].apply(dEspacio)
libros['review/summary'] = libros['review/summary'].apply(dEspacio)
libros['review/text'] = libros['review/text'].apply(dEspacio)

# Crear una nueva columna 'tags' combinando todas las listas de palabras en una sola lista
libros["tags"] = libros['description'] + libros['authors'] + libros['publisher'] + libros['categories'] + libros['review/summary'] + libros['review/text']

# Crear un nuevo DataFrame eliminando las columnas originales que ya no son necesarias
Modelo = libros.drop(columns=["description", "authors", "publisher", "categories", "review/summary", "review/text"])

# Convertir las listas de palabras en la columna 'tags' a cadenas de texto unidas por espacios
Modelo["tags"] = Modelo["tags"].apply(lambda x: " ".join(x))

# Importar CountVectorizer para transformar el texto en vectores de características
from sklearn.feature_extraction.text import CountVectorizer

# Crear una instancia de CountVectorizer con un máximo de 5000 características y eliminar palabras vacías en inglés
cv = CountVectorizer(max_features=5000, stop_words="english")

# Ajustar y transformar los datos de la columna 'tags' en una matriz de características
vector = cv.fit_transform(Modelo["tags"]).toarray()

# Importar cosine_similarity para calcular la similitud coseno entre los vectores de características
from sklearn.metrics.pairwise import cosine_similarity

# Calcular la matriz de similitud coseno entre los vectores
similarity = cosine_similarity(vector)

# Definir una función para recomendar libros basados en el nombre de un libro dado
def recomendacion(nombrelibro):
    # Obtener el índice del libro correspondiente al nombre dado
    index = Modelo[Modelo["Title"] == nombrelibro].index[0]
    
    # Calcular las distancias (similitudes) ordenadas de mayor a menor
    distancias = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
    
    # Imprimir los títulos de los cinco libros más similares, excluyendo el libro dado
    for i in distancias[1:6]:
        print(Modelo.iloc[i[0]].Title)

# Llamar a la función de recomendación con el título de un libro
recomendacion("Its Only Art If Its Well Hung!")
