# REPTE-PIB: Repte de Segmentació Mamogràfica de la Mama

## 📋 Visió General

Aquest repositori conté una solució integral per a la segmentació mamogràfica de la mama utilitzant múltiples enfocaments, incloent tècniques tradicionals de processament d'imatges i aprenentatge profund amb arquitectura U-Net. El projecte implementa pipelines de preprocessament i diversos mètodes de segmentació per extreure regions de teixit mamari d'imatges mamogràfiques.

## 🎯 Objectius del Projecte

- **Preprocessament**: Implementar un pipeline robust de preprocessament per a imatges mamogràfiques
- **Segmentació**: Comparar múltiples enfocaments de segmentació (llindarització d'Otsu, agrupament K-means, U-Net)
- **Avaluació**: Avaluar el rendiment dels diferents mètodes de segmentació
- **Automatització**: Proporcionar un pipeline complet per a la segmentació mamogràfica de la mama

## 📁 Estructura del Repositori

```
REPTE-PIB/
├── README.md                   # Documentació del projecte
├── requirements.txt            # Dependències de Python
├── Repte.pdf                   # Especificació del projecte
├── figures/                    # Conjunt de figures pel projecte
├── code/                       # Codi font
│   ├── preprocessat.ipynb      # Pipeline de preprocessament d'imatges
│   ├── segmentation.ipynb      # Comparació de mètodes de segmentació
│   └── model/                  # Implementació U-Net
│       ├── __init__.py
│       ├── unet_model.py       # Arquitectura U-Net
│       ├── train_unet.ipynb    # Entrenament del model
│       ├── test_unet.ipynb     # Inferència del model
│       └── best_unet.pth       # Pesos del model pre-entrenat
└── data/                       # Conjunt de dades
    ├── train/                  # Conjunt d'entrenament
    │   ├── images/             # Imatges d'entrenament
    │   ├── inputs/             # Imatges d'entrenament preprocessades
    │   └── masks/              # Màscares ground truth
    └── test/                   # Conjunt de test
        ├── images/             # Imatges de test
        ├── inputs/             # Imatges de test preprocessades
        └── masks/              # Màscares generades per diferents mètodes
            ├── kmeans/         # Resultats de segmentació K-means
            ├── otsu/           # Resultats de llindarització d'Otsu
            └── unet/           # Resultats de segmentació U-Net
```

## 🔧 Instal·lació

### Prerequisits
- Python 3.8+
- Gestor de paquets pip

### Configuració de l'Entorn

1. Clona el repositori:
```bash
git clone <url-del-repositori>
cd REPTE-PIB
```

2. Crea un entorn virtual (recomanat):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# o
source venv/bin/activate  # Linux/Mac
```

3. Instal·la les dependències:
```bash
pip install -r requirements.txt
```

## 📚 Dependències

El projecte utilitza les següents llibreries principals:

- **Llibreries Principals**:
  - `numpy` - Computacions numèriques
  - `opencv-python` - Visió per computador i processament d'imatges
  - `matplotlib` - Visualització
  - `scikit-image` - Processament d'imatges avançat
  - `scikit-learn` - Utilitats d'aprenentatge automàtic

- **Aprenentatge Profund**:
  - `torch` - Framework PyTorch
  - `torchvision` - Utilitats de visió per computador per a PyTorch

- **Desenvolupament**:
  - `notebook` - Suport per a Jupyter Notebook
  - `ipykernel` - Kernel IPython per a Jupyter

## 🚀 Ús

### 1. Preprocessament d'Imatges

El pipeline de preprocessament inclou:
- **Eliminació de soroll**: Eliminació de soroll sal i pebre utilitzant filtratge medià
- **Eliminació d'artefactes**: Eliminació d'etiquetes de text i marcadors
- **Estandardització d'orientació**: Posicionament anatòmic consistent
- **Millora del contrast**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
# Executa el notebook de preprocessament
jupyter notebook code/preprocessat.ipynb
```

### 2. Mètodes de Segmentació

S'implementen tres enfocaments de segmentació:

#### Llindarització d'Otsu
- Enfocament ràpid i simple
- Tècnica de llindarització global
- Bo per a imatges d'alt contrast

#### Agrupament K-means
- Enfocament d'agrupament no supervisat
- Separa el teixit del fons
- Mètode sense paràmetres

#### Aprenentatge Profund U-Net
- Segmentació semàntica d'última generació
- Requereix dades d'entrenament
- Enfocament de màxima precisió

```python
# Executa la comparació de segmentació
jupyter notebook code/segmentation.ipynb
```

### 3. Entrenament U-Net

Entrena el model U-Net amb el teu conjunt de dades:

```python
# Entrena el model
jupyter notebook code/model/train_unet.ipynb
```

### 4. Inferència del Model

Genera màscares de segmentació utilitzant la U-Net entrenada:

```python
# Executa la inferència
jupyter notebook code/model/test_unet.ipynb
```

## 🏗️ Arquitectura

### Arquitectura del Model U-Net

La U-Net implementada segueix l'estructura clàssica codificador-decodificador:

- **Codificador**: Camí de contracció amb dobles convolucions i max pooling
- **Coll d'ampolla**: Extracció de característiques a la resolució més baixa
- **Decodificador**: Camí d'expansió amb upsampling i connexions de salt
- **Entrada**: Imatges mamogràfiques en escala de grisos (1 canal)
- **Sortida**: Màscares de segmentació binàries (1 canal)

Característiques clau:
- Canals d'entrada: 1 (escala de grisos)
- Canals de sortida: 1 (màscara binària)
- Arquitectura: 4 nivells de codificador/decodificador
- Connexions de salt per a la preservació de característiques
- Normalització per lots per a l'estabilitat d'entrenament

## 📊 Conjunt de Dades

### Dades d'Entrenament
- **Imatges**: 32 imatges mamogràfiques preprocessades
- **Format**: Fitxers PNG (resolució 768x512)
- **Màscares**: Màscares de segmentació ground truth corresponents
- **Nomenclatura**: Convenció de noms consistent (Original_X_bY_pib.png)

### Dades de Test
- **Imatges**: 4 imatges mamogràfiques de test
- **Format**: Fitxers TIFF i JPG
- **Propòsit**: Avaluació del model i comparació de mètodes

## 🎯 Avaluació del Rendiment

El projecte compara els mètodes de segmentació basant-se en:
- **Precisió**: Precisió de classificació píxel a píxel
- **Coeficient de Dice**: Solapament entre màscares predites i ground truth
- **IoU (Intersection over Union)**: Índex de Jaccard per a la qualitat de segmentació
- **Temps de Processament**: Eficiència computacional

## 🔬 Metodologia

### Pipeline de Preprocessament
1. **Filtratge Medià**: Eliminar soroll sal i pebre (kernel 3x3)
2. **Eliminació d'Artefactes**: 
   - Retalla marges i elimina etiquetes de text
   - Utilitza operacions morfològiques i inpainting
3. **Estandardització d'Orientació**: 
   - Detecta la distribució del teixit
   - Gira horitzontalment si cal per consistència
4. **Millora del Contrast**: 
   - Normalitza el rang d'intensitat [0, 255]
   - Aplica CLAHE per a millora local del contrast

### Enfocaments de Segmentació
1. **Mètodes Tradicionals**:
   - Llindarització d'Otsu amb post-processament morfològic
   - Agrupament K-means per a separació teixit/fons
   
2. **Aprenentatge Profund**:
   - Entrenament U-Net amb conjunt de dades personalitzat
   - Augmentació de dades i divisió de validació
   - Avaluació del model i inferència


## 📄 Llicència

Aquest projecte està desenvolupat amb fins acadèmics. Si us plau, consulta el document d'especificació del projecte (`Repte.pdf`) per obtenir requisits i directrius detallades.

## 🔗 Referències

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Documentació d'OpenCV
- Documentació de PyTorch
- Documentació de Scikit-image
