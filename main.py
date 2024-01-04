from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from pandasgui import show
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


k = 3 # - Determina o numero de clusters
n_pca = 2 # - Determina o numero de colunas do PCA
normalizador = MinMaxScaler() # - Normalizador
pca = PCA(n_components=n_pca) # - PCA

def CarregaDados():
    #Carregando dados
    dados = pd.read_csv('entradas_breast.csv')
    labels = pd.read_csv('saidas_breast.csv')

    #Normalizando e Aplicando PCA para compressão e evitar correlação
    dados = normalizador.fit_transform(dados)
    dados = pca.fit_transform(dados)

    return dados, labels

# - Métricas De Validação -

# Elbow - Encontrar o melhor valor de K através da curva 'Cutuvelo'
# ARI - Encontrar o melhor valor de K através de labels ja conhecidos (Maior Score = Melhor)
# SR - Encontrar o melhor valor de K através de labels desconhecidos (Maior Score = Melhor)

def elbow():
    #Carregando dados
    dados, labels = CarregaDados()

    #Variaveis de valores de K e Score
    valores_k = []
    inertias = []

    #Armazenando score para cada valor de K em um intervalo do 1 a 15
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(dados)
        valores_k.append(i)
        inertias.append(kmeans.inertia_)

    #Visualizando grafico
    plt.plot(valores_k, inertias)
    plt.title('Validação de ELbow')
    plt.xlabel('Valores de K')
    plt.ylabel('Inertia')
    plt.show()

def validacao_ARI():
    #Carregando dados
    dados, labels = CarregaDados()

    #Re-estruturando label (n,)
    labels = np.reshape(labels, [labels.shape[0]])

    #Variaveis de valores de K e Score
    valores_k = []
    ARI = []

    #Armazenando score para cada valor de K em um intervalo do 1 a 15
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(dados)
        valores_k.append(i)
        ARI.append(adjusted_rand_score(labels, kmeans.labels_))

    #Visualizando grafico
    plt.plot(valores_k, ARI)
    plt.title('Validação ARI')
    plt.xlabel('Valores de K')
    plt.ylabel('ARI Score')
    plt.show()

def validacao_SR():
    #Carregando dados
    dados, labels = CarregaDados()

    #Variaveis de valores de K e Score
    valores_k = []
    SR = []

    #Armazenando score para cada valor de K em um intervalo do 2 a 15 (K = 1 gera erro)
    for i in range(2,15):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(dados)
        valores_k.append(i)
        SR.append(silhouette_score(dados, kmeans.labels_))

    #Visualizando grafico
    plt.plot(valores_k, SR)
    plt.title('Validação SR')
    plt.xlabel('Valores de K')
    plt.ylabel('SR Score')
    plt.show()


def Cleturing():
    #Carregando dados
    dados, labels = CarregaDados()

    #Clusterizando dados
    kmeans = KMeans(n_clusters=k, random_state=0).fit(dados)

    #Atribuindo clusterização aos dados
    cluster_map = pd.read_csv('entradas_breast.csv')
    cluster_map['Cluster'] = kmeans.labels_

    #Definindo limites do eixo X e Y do grafico
    x_min, x_max = dados[:,0].min() - 0.3, dados[:,0].max() + 0.3
    y_min, y_max = dados[:,1].min() - 0.3, dados[:,1].max() + 0.3

    #Definindo centroids
    centroids = kmeans.cluster_centers_

    #Plotando grafico com dados e centroids
    plt.scatter(dados[:,0], dados[:,1], c=kmeans.labels_)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=169, linewidths=3, color='r', zorder=8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    #Exebindo dados com coluna de clusterização
    show(cluster_map)

elbow()
validacao_ARI()
validacao_SR()
Cleturing()


