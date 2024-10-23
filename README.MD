# Classification de textes et images sur les produits Rakuten

Ce projet contient les notebooks et fichiers relatifs à un projet dans le cadre d'une formation assurée par DataScientest.
L'objectif du projet est de développer des modèles pour classifier automatiquement des produits du catalogue Rakuten à partir d'éléments textuels
(désignation et, optionnellement, description) et de photos correspondantes.

## Auteurs et contexte

Le projet a été réalisée par Saliou Diedhiou, Eren Ustundag, Luc Jamet et Leila Ruiz sous la supervision d'Emilie Greff.
Il s'agit d'un travail de formation au métier de data scientist, formation continue débutée en novembre 2022.

# Installation

## Ajustement des répertoires-parents

Les notebooks contenus dans ce projet ont été conçus sur un espace partagé qui a requis de définir un répertoire-parent. Pour fonctionner sur votre
espace de travail, il sera nécessaire de modifier les chemins. Dans la plupart des cas, il devrait suffire de modifier la cellule où se trouve la
déclaration de la variable `DRIVE_PATH`, cellule qui figurera parmi les premières de chaque notebook :

- remplacer la valeur assignée à `DRIVE_PATH` par l'emplacement de votre copie du projet
- si vous travaillez sans Google Colab, retirer la ligne `from google.colab import drive`

## Installation de librairies

Il est supposé que votre environnement de travail inclut déjà les librairies les plus habituelles de la Data Science, comme NumPy, Pandas, Matplotlib,
Seaborn ou encore Scikit-Learn. Les notebooks sont en prime susceptibles de tenter l'installation de librairies plus spécialisées. En cas d'erreur de
type `ModuleNotFoundError`, merci de procéder manuellement à l'installation du ou des modules manquants.

## Fichier-source avec textes traduits

Le fichier-source ayant servi à l'entraînement et à la validation des modèles est `CSV_avec_traduction/df_avec_traduction.csv`. Il est livré dans une
version compressée avec le projet, et il est nécessaire de le décompresser avant utilisation.

## Vecteurs pré-entraînés pour FastText

Certains des modèles FastText (text mining) que nous avons entraînés s'appuient sur un ensemble de vecteurs pré-entraînés. Le fichier qui les contient
étant lourd, il ne peut être versionné dans le projet Git et il est nécessaire de le construire soi-même. Pour ceci :

- [télécharger](https://fasttext.cc/docs/en/crawl-vectors.html) le fichier au format `bin` pour la langue française sur le site de FastText 
- le placer dans `models/fasttext_files` et le décompresser
- se rendre dans ce meme répertoire et y exécuter le script Python `reduce_fasttext_vec_dim.py`

## Images

Le volume total représenté par les images est excessif pour une sauvegarde dans le projet GitHub. Il vous faudra les télécharger depuis le site du
[Rakuten Data Challenge](https://challengedata.ens.fr/challenges/35). De plus, il est probable que la disposition des images sur votre poste diffère
de celle sur laquelle nous avons travaillé, aussi vous faudra-t-il adapter le contenu de la colonne `image_path` des dataframes (voir la section
"Fichiers CSV ci-dessous"). La logique d'appellation des images est la suivante :

```
image_<imageid>_product_<productid>.jpg
```

en référence aux colonnes `imageid` et `productid` des fichiers CSV fournis.

# Contenu

## Notebooks

### Pré-traitement et visualisation des données

Le notebook correspondant à cette partie est `preprocessing_dataviz/preprocessing_and_dataviz.ipynb`. Il est découpé en quelques sections :

- *Pré Requis* - Installation de librairies et fichiers associés
- *Import des fichiers* - Chargement du dataframe-source, résultat d'une adaptation du dataframe fourni par Rakuten à nos besoins (ajout
d'une colonne pour pointer vers les images)
- *Exploration des données* - Premier aperçu des données : volumétrie, données manquantes, distribution des classes-cibles
- *Phase de Pre Processing* - Assignation de libellés clairs aux classes-cibles, correction des textes et balises HTML, traduction des
données textuelles non françaises
- *Représentations graphiques* - Construction de diagrammes pour mieux prendre en main les données et anticiper les difficultés.

### Text mining

La modélisation sur la base des textes se trouve dans le notebook `text_mining/text_mining_models.ipynb`. Il est agencé en plusieurs sections et
sous-sections. Les sections principales sont :

- *Pré-requis : installations et import de modules* - Import de librairies spécialisées
- *Import des fichiers* - Chargement des données
- *Pré-traitement : mise en minuscules et retrait des mots-stops* - Pré-traitement requis pour l'entraînement et l'exploitation de différents modèles
- *Découpage en données d'entraînement et données de test* - Assurer un découpage entre jeu d'entraînement et jeu de test qui soit homogène d'une
phase de modélisation à une autre
- *Définition de fonctions d'affichage et d'analyse* - Déclaration de quelques fonctions utiles au travail de modélisation, en particulier pour
évaluer la qualité des modèles
- *Essai de différents classificateurs (modélisation "classique")* - Exploration, calage et évaluation de modèles classiques de machine learning
(p. ex. Random Forest)
- *Performances des classificateurs "classiques" sur les différentes catégories de produits* - Evaluation plus détaillée des modèles classiques de
machine learning
- *Essai de FastText* - Exploration du modèle FastText, en principe plus puissant que le machine learning classique
- *Essai Word2Vec* - Modèles de machine learning et de deep learning en s'appuyant sur la tokenisation avancée offerte par Word2Vec
- *Essai CamemBERT avec TextCNN* - Modèle de deep learning s'appuyant sur la tokenisation via CamemBERT, version francophone de BERT

### Computer vision

Le travail de modélisation à partir des images est disponible dans le notebook `computer_vision/computer_vision_modeling.ipynb`. La section d'intérêt
est *Classification des images*. Dans celle-ci, les premières cellules assurent l'installation des librairies requises puis le chargement du dataframe
qui contient le chemin vers les images ainsi que les classes-cibles (catégories des produits). S'ensuivent différentes sous-sections, chacune dédiée à
un modèle de deep learning appliqué à des images :

- *Classification des images avec le modèle Inception de Keras*
- *Classification des images avec le modèle Xception de Keras*
- *Classification des images avec un CNN classique de Keras*
- *Classsification des images avec un CNN Resnet152 de Keras*

## Modèles

L'entraînement de certains modèles peut être long, c'est pourquoi des sauvegardes sont disponibles :

- des modèles entraînés ont été enregistrés dans le répertoire `models/saved_models`
- des résultats de calcul ont été placés dans `models/models_results`

Le contenu n'est toutefois pas exhaustif car les modèles de deep learning étaient trop massifs pour pouvoir être ajoutés au dépôt. Il sera
indispensable de ré-entraîner ces derniers localement.

## Données-sources

### Fichiers CSV

Plusieurs fichiers CSV contenant les données d'entrées sont disponibles :

- dans le répertoire `train_test/csv_files`, les deux fichiers originaux fournis par Rakuten et contenant les données d'entraînement,
l'un avec les variables explicatives et données nécessaires à l'identification des images (`X_train_with_img_path.csv`), l'autre contenant
les catégories des produits (variable-cible, fichier `Y_train_CVw08PX.csv`)
- dans le répertoire `CSV_avec_traduction`, le fichier `df_avec_traduction_test.csv` contenant le résultat du pré-traitement et servant
de point de départ à l'entraînement et à la validation des modèles

# Utilisation

Les notebooks ont été conçus pour fonctionner indépendamment les uns des autres. Chacun d'entre eux est découpé en sections, les premières
servant à charger les modules Python et les données à explorer et, le cas échéant, déclarer quelques fonctions utiles.

Certains "checkpoints" ont aussi été déclarés pour pouvoir charger rapidement des modèles ou résultats au sein de certaines sections des
notebooks. Ils ne seront bien sûr utilisables qu'avec les données que nous avons pu inclure au projet Git.

# Scores obtenus

Voici un résumé des scores obtenus avec les principaux modèles :

| Domaine         | Modèle                                 | Métrique            | Score - Entraînement | Score - Test  |
|-----------------|----------------------------------------|---------------------|----------------------|---------------|
| Text mining     | TF-IDF + Stochastic gradient descent   | Accuracy équilibrée | 79%                  | 73%           |
| Text mining     | FastText                               | Accuracy équilibrée | 90%                  | 77%           |
| Text mining     | Word2Vec + Réseau dense                | Accuracy            | 80%                  | 72%           |
| Text mining     | CamemBERT + TextCNN                    | Accuracy            | 78%                  | 84%           |
| Computer vision | Inception                              | Accuracy            | 72%                  | 62%           |
| Computer vision | Xception                               | Accuracy            | 54%                  | 52%           |
| Computer vision | Resnet152                              | Accuracy            | 53%                  | 52%           |
| Multimodal      | Voteur : CamemBERT + Inception         | Accuracy équilibrée | -                    | 81%           |