# DAA_ML - Machine Learning for Chemical Reaction Prediction

This repository contains machine learning models and analyses for predicting chemical reaction outcomes, specifically focusing on:
- Suzuki-Miyaura coupling reactions
- Buchwald-Hartwig C-N coupling reactions  
- HOMO-LUMO energy predictions

## 📁 Repository Structure

```
DAA_ML/
├── notebooks/          # Jupyter notebooks with ML models and analysis
├── data/              # Datasets for training and evaluation
└── README.md          # This file
```

## 📓 Notebooks

### Suzuki-Miyaura Coupling Reactions
- **`Suzuki_Coupling_DataSet_Combined_LGBM.ipynb`** - LightGBM model on combined Suzuki-Miyaura datasets
- **`Suzuki_Coupling_Dataset_Combined_NN.ipynb`** - Neural Network model on combined datasets
- **`Suzuki_Coupling_datasetC.ipynb`** - Analysis and modeling on Dataset C
- **`Suzuki_Coupling_datasetD_LGBM.ipynb`** - LightGBM model on Dataset D

### Buchwald-Hartwig C-N Coupling Reactions
- **`C_N_C.ipynb`** - Main C-N coupling reaction prediction model
- **`C_N_C_feature_selection.ipynb`** - Feature selection and engineering for C-N coupling

### Molecular Properties
- **`Avalon_Fingerprint_For_HOMO_LUMO_ML.ipynb`** - HOMO-LUMO energy prediction using Avalon fingerprints

## 📊 Datasets

### Suzuki-Miyaura Coupling Data
Located in `data/` directory with the prefix `S_M_`:
- `S_M_DataSet2.csv/xlsx` - Dataset 2
- `S_M_Data_SetA.csv/xlsx` - Dataset A
- `S_M_DatasetB.csv/xlsx` - Dataset B
- `S_M_DatasetC.csv` - Dataset C
- `S_M_DatasetD.csv` - Dataset D
- `S_M_Dataset_Combined.csv/xlsx` - Combined dataset from multiple sources

### Buchwald-Hartwig C-N Coupling Data
- `Buchwald_yield_data.csv` - Yield data for Buchwald-Hartwig reactions

### Additional Data
- `aap9112_data_file_s1.xlsx` - Supplementary data file

## 🚀 Getting Started

### Prerequisites

The notebooks require the following Python packages:
- pandas
- numpy
- scikit-learn
- lightgbm (for LGBM models)
- tensorflow or pytorch (for neural network models)
- rdkit (for molecular fingerprints and chemical calculations)
- matplotlib/seaborn (for visualizations)

### Installation

```bash
# Clone the repository
git clone https://github.com/dakuamoah/DAA_ML.git
cd DAA_ML

# Install dependencies (recommended: use a virtual environment)
pip install pandas numpy scikit-learn lightgbm tensorflow rdkit matplotlib seaborn jupyter
```

### Usage

1. Open Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory

3. Open any notebook and run the cells sequentially

## 📈 Models and Approaches

### Machine Learning Algorithms Used:
- **LightGBM (LGBM)**: Gradient boosting framework for regression/classification tasks
- **Neural Networks (NN)**: Deep learning models for complex pattern recognition
- **Feature Engineering**: Molecular fingerprints (Avalon) and domain-specific features

### Prediction Tasks:
- **Reaction Yield Prediction**: Predicting the percentage yield of coupling reactions
- **Molecular Property Prediction**: HOMO-LUMO energy gap calculations
- **Feature Importance Analysis**: Understanding key factors affecting reaction outcomes

## 🔬 Research Context

This work focuses on using machine learning to predict outcomes of important organic chemistry reactions:

- **Suzuki-Miyaura Coupling**: A palladium-catalyzed cross-coupling reaction widely used in pharmaceutical synthesis
- **Buchwald-Hartwig Coupling**: A palladium-catalyzed C-N bond formation reaction critical for drug discovery
- **HOMO-LUMO Analysis**: Predicting frontier molecular orbital energies for understanding reactivity

## 📝 Notes

- The datasets include various molecular descriptors, reaction conditions, and experimental yields
- Multiple file formats (CSV, XLSX) are provided for compatibility
- Some datasets have both individual and combined versions for different modeling approaches

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

## 📄 License

Please check with the repository owner for licensing information.
