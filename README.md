# TokenFormer - Réimplémentation

Ce projet vise à réimplémenter le papier **"TOKENFORMER: RETHINKING TRANSFORMER SCALING WITH TOKENIZED MODEL PARAMETERS"**. Il explore une approche alternative au scaling des Transformers en optimisant la représentation des paramètres du modèle sous forme tokenisée.

## 📌 Contenu du projet

### 📂 `models/`
Ce dossier contient l'implémentation des architectures utilisées dans le projet.  
- `tokenformer.py` : Implémentation du modèle TokenFormer.
- `transformer.py` : Implémentation d'un Transformer standard.
- `self_attention.py` : Module de self-attention pour le TokenFormer.
- `pattention.py` : Module de Pattention (une variante de l'attention).

### 📂 `data/`
Ce dossier est dédié à la gestion des données.  
- `dataset.py` : Définit les fonctions de chargement et de prétraitement des données.

### 📂 `training/`
Ce dossier regroupe les fonctions liés à l'entraînement des modèles.  
- `train.py` : Contient les fonctions pour :
  - Entraîner un modèle Transformer standard.
  - Entraîner un modèle TokenFormer.
  - Entraîner un modèle avec **scaling progressif**.

### 📂 `checkpoints/`
Répertoire dédié à la sauvegarde des modèles entraînés sous forme de checkpoints.

### 📂 `utils/`
Contient d'autres expérimentations altérnatives.

### 📂 `other/`
Ce dossier contient des expérimentations et des versions intermédiaires du projet. Il inclut :
- `tests_version0/` : Tests initiaux du modèle.
- `training_version0/` : Versions de test des scripts d'entraînement.
- `models_version0/` : Versions préliminaires des modèles.

## 👨‍💻 Auteurs
Ce projet a été réalisé par :
- **Hocine Kadem**
- **Huang Tian**
- **Kande Seydina**

📍 **Sorbonne Université**

## 📜 Références
- **TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters**  
  _[https://arxiv.org/abs/2410.23168]_

