"""
clustering of word embeddings

@TODO documentation of the module
"""
import numpy as np

from sklearn.base import BaseEstimator
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import umap
from matplotlib import pyplot as plt

class WordClustering(BaseEstimator):
    """ theme-affinity vectorization of documents

    w2v_size : int, default=128
       size of the hidden layer in the embedding Word2Vec model

    n_clusters : int, default=30
        number of clusters,  to the number of output parameters for the
        vectorization.
        It is advised to set `n_clusters` to the approximate number of
        lexical fields

    clustering : sklearn.cluster instace, default=KMeans(n_clusters=30)
        clustering algorithm
        The number of clusters must be equal to `n_clusters`

    pretrained : bool, default=False
        False to train a new w2v model
        True to use a model already trained

    model_path : str, default=None
        path to the trained w2v model
        Only used when `pretrained` is set to True

    """
    def __init__(self,
                 w2v_size=128,
                 n_clusters=30,
                 clustering=KMeans(n_clusters=30),
                 pretrained=False,
                 model_path=None):
        self.w2v_size = w2v_size
        self.n_clusters = n_clusters
        self.clustering = clustering
        self.pretrained = pretrained
        self.model_path = model_path

        # vocabulary
        self.vocabulary_ = None
        # distribued representation of the words
        self.word_vectors_ = None
        # cluster id for each word
        self.cluster_ids_ = None
        
        # word embedded
        self.embedded = None
        self.clustering.set_params(n_clusters=n_clusters)


    def fit(self, X=None, y=None, **fit_params):
        """ train w2v and clustering models

        Parameters
        ----------
        X : iterable of iterable, defaul=None
            corpus of tokenized documents if `pretrained`=False
            else, X=None and the pretrained model is used


        y : None

        fit_params : additionnal parameters for word2vec algorithm

        Returns
        -------
        self

        """
        
        if self.pretrained:
            w2v = KeyedVectors.load(self.model_path)
        else:
            w2v = Word2Vec(X, size=self.w2v_size)
        
        
        self.vocabulary_ = w2v.wv.vocab
       
        self.word_vectors_ = w2v[self.vocabulary_]
        
        
        reducer = umap.UMAP(metric='cosine',low_memory=True, random_state=42)
        self.embedded = reducer.fit_transform(self.word_vectors_)
        
        X_train, X_test= train_test_split(self.embedded,
                             test_size=0.8, random_state=42) 
        
        self.clustering.fit(X_train)
        labels = self.clustering.labels_

        KN = KNeighborsClassifier(n_neighbors=5)
        KN.fit(X_train,labels)
        self.cluster_ids_ = KN.predict(self.embedded)
        print("oui")
        """
        self.clustering.fit(self.word_vectors_)
        self.cluster_ids_ = self.clustering.labels_
        print("oui")
        """
        return self
    
    def display_one_cluster(self,num_cluster):
        """display clusters with their numbers"""
        
        print(self.embedded)
        embedded_DF=pd.DataFrame(data=self.embedded ,columns=['x','y'])
        embedded_DF['labels']=self.cluster_ids_
        embedded_DF=embedded_DF.groupby('labels').mean()
        embedded_DF['labels']=set(self.cluster_ids_)
        select=embedded_DF[embedded_DF['labels']==num_cluster]
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(self.embedded[:,0],self.embedded[:,1], alpha=0.2,
                             c=plt.cm.nipy_spectral(self.cluster_ids_/300))

        def label_point(x, y, val, ax):
            
            ax.text(x, y, str(int(num_cluster)))
     

        label_point(select.x, select.y, num_cluster, ax)

        ax.set_title('Agglomerative Clustering')
        plt.colorbar(scatter)
        plt.savefig('Clustering2')
        
    def display_clusters(self):
        """display clusters with their numbers"""

        print(self.embedded)
        embedded_DF=pd.DataFrame(data=self.embedded ,columns=['x','y'])
        embedded_DF['labels']=self.cluster_ids_
        embedded_DF=embedded_DF.groupby('labels').mean()
        embedded_DF['labels']=set(self.cluster_ids_)
        print(embedded_DF)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(self.embedded[:,0],self.embedded[:,1], alpha=0.2,
                             c=plt.cm.nipy_spectral(self.cluster_ids_/300))

        def label_point(x, y, val, ax):
            cpt=0
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                cpt+=1
                ax.text(point['x'], point['y'], str(int(point['val'])))
            print(cpt)

        label_point(embedded_DF.x, embedded_DF.y, embedded_DF.labels, ax)

        ax.set_title('Agglomerative Clustering')
        plt.colorbar(scatter)
        plt.savefig('Clustering')


    def transform(self, X, y=None):
        """ transforms each row of `X` into a vector of clusters affinities

        Parameters
        ----------
        X : iterable of iterable

        y: None

        Returns
        -------
        numpy.ndarray, shape=(n, p)
            transformed docments, where `p=n_cluster`

        """
        vectors = []

        for x in X:
            vector = np.zeros(self.n_clusters)
            count = 0
            for t in x:
                try:
                    word_id = self.vocabulary_[t].index
                    word_cluster = self.cluster_ids_[word_id]
                    vector[word_cluster] = vector[word_cluster] + 1
                    count += 1
                # except word is not in vocabular
                except KeyError:
                    pass
            if count > 0:
                vectors.append(vector / count)
            else:
                vectors.append(vector)

        return np.array(vectors)

    def get_clusters_words(self):
        """ return the words in each cluster


        Returns
        -------
        dict
            keys are cluster ids, values are lists of words

        """
        words_cluster = {}
        for cluser_id in np.unique(self.cluster_ids_):
            words_cluster[str(cluser_id)] = []

        for i, word in enumerate(self.vocabulary_):
            label = str(self.cluster_ids_[i])
            words_cluster[label].append(word)

        return words_cluster




def embed_corpus(X, n_clusters, clustering, **kwargs):
    """ transforms X into vector of cluster affinities

    ..deprecated use `WordClustering` object instead
    Parameters
    ----------
    X : iterable of iterable, (length=n)
        corpus of document

    clustering : sklearn.cluster object
        instanciated clustering algorithm

    Returns
    -------
    np.ndarray, shape=(n, n_clusters)

    """
    # fit
    w2v = Word2Vec(X, size=128)

    words = w2v.wv.vocab
    word_vectors = w2v[words]

    pca_word_vectors = PCA(n_components=0.9).fit_transform(word_vectors)

    # clustering = AgglomerativeClustering(n_clusters, affinity='euclidean')

    cluster_ids = clustering.fit_predict(pca_word_vectors)
    
    # transform
    vectors = []
    for x in X:
        vector = np.zeros(n_clusters)
        count = 0
        for t in x:
            try:
                word_id = words[t].index
                word_cluster = cluster_ids[word_id]
                vector[word_cluster] = vector[word_cluster] + 1
                count += 1
            except KeyError:
                pass
        vectors.append(vector / count)

    return np.array(vectors), cluster_ids
