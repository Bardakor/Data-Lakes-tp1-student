import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
import os
from collections import OrderedDict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d'un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d'entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the data
    print('Loading Data')
    data = pd.read_csv(data_file)

    # Step 2: Handle missing values
    print('Cleaning Data')
    data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    print('Encoding Labels')
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(
        data['family_accession'])

    # Save the label encoder
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(label_encoder, encoder_path)

    # Save the label mapping to a text file
    mapping_path = os.path.join(output_dir, 'label_mapping.txt')
    with open(mapping_path, 'w') as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f"{label}\t{i}\n")

    # Step 4: Initialize lists for train/dev/test indices
    train_indices = []
    dev_indices = []
    test_indices = []

    # Group data by class and distribute according to size
    print("Distributing data")
    class_groups = data.groupby('class_encoded')

    for cls in tqdm.tqdm(range(len(label_encoder.classes_))):
        if cls not in class_groups.groups:
            continue

        indices = class_groups.groups[cls]
        count = len(indices)
        indices = indices.to_numpy()

        if count == 1:
            # Single sample goes to test
            test_indices.extend(indices)
        elif count == 2:
            # Two samples split between dev and test
            dev_indices.append(indices[0])
            test_indices.append(indices[1])
        elif count == 3:
            # Three samples split across all sets
            train_indices.append(indices[0])
            dev_indices.append(indices[1])
            test_indices.append(indices[2])
        else:
            # For larger classes, do stratified split
            temp_train, temp_test = train_test_split(
                indices, test_size=0.2, random_state=42)
            temp_train, temp_dev = train_test_split(
                temp_train, test_size=0.2, random_state=42)
            train_indices.extend(temp_train)
            dev_indices.extend(temp_dev)
            test_indices.extend(temp_test)

    # Step 5: Convert index lists to numpy arrays
    train_indices = np.array(train_indices)
    dev_indices = np.array(dev_indices)
    test_indices = np.array(test_indices)

    # Step 6: Create DataFrames from the selected indices
    train_data = data.iloc[train_indices].copy()
    dev_data = data.iloc[dev_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Step 7: Drop unused columns
    columns_to_keep = ['sequence', 'family_accession', 'class_encoded']
    train_data = train_data[columns_to_keep]
    dev_data = dev_data[columns_to_keep]
    test_data = test_data[columns_to_keep]

    # Step 8: Save train/dev/test datasets as CSV
    print("Saving processed datasets")
    train_data.to_csv(os.path.join(
        output_dir, 'train_processed.csv'), index=False)
    dev_data.to_csv(os.path.join(output_dir, 'dev_processed.csv'), index=False)
    test_data.to_csv(os.path.join(
        output_dir, 'test_processed.csv'), index=False)

    # Step 9: Calculate class weights from the training set
    class_counts = train_data['class_encoded'].value_counts()
    total_samples = len(train_data)
    n_classes = len(label_encoder.classes_)

    # Step 10: Normalize weights and scale
    class_weights = {
        cls: total_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }

    # Handle classes that might not be in training set
    for cls in range(n_classes):
        if cls not in class_weights:
            class_weights[cls] = total_samples / \
                n_classes  # Assign average weight

    # Step 11: Save the class weights
    weights_path = os.path.join(output_dir, 'class_weights.txt')
    with open(weights_path, 'w') as f:
        for cls in sorted(class_weights.keys()):
            f.write(f"{cls}\t{class_weights[cls]}\n")

    print("Preprocessing completed successfully!")
    print(f"Train set size: {len(train_data)}")
    print(f"Dev set size: {len(dev_data)}")
    print(f"Test set size: {len(test_data)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str,
                        required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
