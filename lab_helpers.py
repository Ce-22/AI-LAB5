"""
Módulo auxiliar de funções para o Laboratório 5 de IA/ML (Aprendizado Não Supervisionado e Modelos de Cluster).
Autores: Cesar Gritti e Patrick Barreira (PUC-SP)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)


def load_and_preprocess_customer_data(csv_path: str = "data/customer_personality_analysis.csv"):
    """
    Carrega o dataset Customer Personality Analysis do Kaggle, realiza limpeza,
    imputação de dados faltantes, engenharia de atributos e redução com PCA.
    """
    # 1. Carregamento com separador tabular
    df_raw = pd.read_csv(csv_path, sep='\t')
    df = df_raw.copy()

    # 2. Remoção de colunas constantes sem variância
    cols_to_drop = ['ID']
    for col in ['Z_CostContact', 'Z_Revenue']:
        if col in df.columns:
            cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 3. Imputação de renda (Income) pela mediana do nível de escolaridade
    df['Income'] = df.groupby('Education')['Income'].transform(lambda s: s.fillna(s.median()))

    # 4. Remoção de outliers extremos óbvios (anos de nascimento antes de 1920 e renda > 600.000)
    df = df[df['Year_Birth'] >= 1920].copy()
    df = df[df['Income'] < 600000].copy()

    # 5. Engenharia de atributos
    ref_year = 2026
    df['Age'] = ref_year - df['Year_Birth']

    # Gastos totais em categorias de produtos
    spending_cols = [
        'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
    ]
    df['Total_Spent'] = df[spending_cols].sum(axis=1)

    # Compras totais por canal
    purchase_cols = [
        'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumDealsPurchases'
    ]
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

    # Estrutura familiar
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Has_Children'] = (df['Children'] > 0).astype(int)

    # Status civil simplificado e tamanho familiar estimado
    partner_map = {
        'Married': 2, 'Together': 2,
        'Single': 1, 'Divorced': 1, 'Widow': 1,
        'Alone': 1, 'Absurd': 1, 'YOLO': 1
    }
    df['Adults'] = df['Marital_Status'].map(partner_map).fillna(1).astype(int)
    df['Family_Size'] = df['Adults'] + df['Children']

    # Antiguidade do cliente em dias
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    max_date = df['Dt_Customer'].max()
    df['Customer_Days'] = (max_date - df['Dt_Customer']).dt.days

    # Engajamento em campanhas de marketing
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    df['Total_Accepted_Cmp'] = df[cmp_cols].sum(axis=1)

    # Razões de canais
    df['Web_Purchase_Share'] = df['NumWebPurchases'] / (df['Total_Purchases'] + 1e-5)
    df['Store_Purchase_Share'] = df['NumStorePurchases'] / (df['Total_Purchases'] + 1e-5)

    # Seleção de atributos numéricos principais para modelagem de clustering
    feature_cols = [
        'Income', 'Age', 'Total_Spent', 'Total_Purchases',
        'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth',
        'Children', 'Customer_Days', 'Total_Accepted_Cmp'
    ]

    X = df[feature_cols].copy()

    # Normalização robusta com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return df_raw, df, feature_cols, X_scaled, scaler, pca, X_pca


def compute_clustering_metrics(X, labels, model_name: str = "Modelo"):
    """
    Calcula Silhouette Score, Davies-Bouldin Index e Calinski-Harabasz Score.
    Trata amostras de ruído (rótulo -1 no DBSCAN).
    """
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = np.sum(labels == -1)

    if n_clusters < 2:
        return {
            'Modelo': model_name,
            'N_Clusters': n_clusters,
            'Ruído': n_noise,
            'Silhouette': np.nan,
            'Davies-Bouldin': np.nan,
            'Calinski-Harabasz': np.nan
        }

    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])

    return {
        'Modelo': model_name,
        'N_Clusters': n_clusters,
        'Ruído': n_noise,
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'Calinski-Harabasz': round(ch, 2)
    }


def evaluate_kmeans_range(X, k_min: int = 2, k_max: int = 10, random_state: int = 42):
    """
    Avalia K-Means variando k no intervalo [k_min, k_max] calculando Inércia e Silhouette.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []
    db_scores = []
    ch_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
        db_scores.append(davies_bouldin_score(X, km.labels_))
        ch_scores.append(calinski_harabasz_score(X, km.labels_))

    metrics_df = pd.DataFrame({
        'k': k_values,
        'Inertia': inertias,
        'Silhouette': silhouettes,
        'Davies_Bouldin': db_scores,
        'Calinski_Harabasz': ch_scores
    })

    return metrics_df


def summarize_cluster_profiles(df: pd.DataFrame, cluster_labels: np.ndarray, cluster_col: str = 'Cluster'):
    """
    Gera tabela de perfil analítico comparando as médias de atributos-chave por cluster.
    """
    df_temp = df.copy()
    df_temp[cluster_col] = cluster_labels

    metrics_to_show = [
        'Age', 'Income', 'Total_Spent', 'Total_Purchases',
        'NumDealsPurchases', 'NumWebPurchases', 'NumStorePurchases',
        'NumCatalogPurchases', 'NumWebVisitsMonth', 'Recency',
        'Children', 'Family_Size', 'Total_Accepted_Cmp'
    ]

    profile = df_temp.groupby(cluster_col)[metrics_to_show].mean().round(2)
    counts = df_temp[cluster_col].value_counts().rename('Total_Clientes')
    percentages = (df_temp[cluster_col].value_counts(normalize=True) * 100).round(1).rename('Share_%')

    summary = pd.concat([counts, percentages, profile], axis=1).sort_index()
    return summary
