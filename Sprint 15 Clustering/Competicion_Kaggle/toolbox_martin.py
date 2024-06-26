# TOOLBOX MARTIN

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
IMPORTAR

import bootcampviztools as bt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import toolbox_martin as tbm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.metrics import ConfusionMatrixDisplay,balanced_accuracy_score, classification_report, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

"""

def analizar_df(dataframe):
    """dataframe: estructura de datos en formado tabular de tipo pandas dataframe
     Devuelve otro dataframe con el numero de columnas, tipo de cada una, cardinalidad y porcentaje de NaN """
    # Inicializar listas para almacenar información
    columnas = []
    dtypes = []
    cardinalidades = []
    porcentajes = []
    nulos = []
    porcentajes_nulos = []

    # Iterar sobre las columnas del DataFrame
    for col in dataframe.columns:

        # Obtener el tipo de dato de la columna
        dtype = dataframe[col].dtype

        # Obtener la cardinalidad y porcentaje
        unique_values = dataframe[col].nunique()
        total_values = len(dataframe[col])
        porcentaje = (unique_values / total_values) * 100
        nulos_columna = dataframe[col].isnull().sum()
        porcentaje_nulos = (nulos_columna / total_values) * 100

        # Almacenar la información en las listas
        columnas.append(col)
        dtypes.append(dtype)
        cardinalidades.append(unique_values)
        porcentajes.append(porcentaje)
        nulos.append(nulos_columna)
        porcentajes_nulos.append(porcentaje_nulos)

    # Crear un DataFrame con la información recopilada
    resultados = pd.DataFrame({
        'Columna': columnas,
        'Tipo': dtypes,
        'Cardinalidad': cardinalidades,
        '% Cardinalidad': porcentajes,
        'Numero de nulos':nulos,
        '% Nulos': porcentajes_nulos 
        
    })

    return resultados


def barras_frecuencias(df, columna, mostrar_valores=False, giro=90, relativas=False, tamaño=False):
    # Montamos el cuadro de 2 figuras por fila si se requieren frecuencias relativas
    if relativas:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Frecuencias absolutas
    if relativas:
        ax_absolutas = axes[0]
    else:
        ax_absolutas = ax

    # Tamaño de la serie
    serie_absolutas = df[columna].value_counts()  # Acotamos el numero de barras si queremos
    
    if tamaño != False and type(tamaño) == int:
        serie_absolutas = serie_absolutas.head(tamaño)
    
    sns.barplot(x=serie_absolutas.index, y=serie_absolutas, ax=ax_absolutas, palette='crest', hue=serie_absolutas.index, legend=False)
    ax_absolutas.set_ylabel('Frecuencia')
    ax_absolutas.set_title(f'Distribución de {columna}',pad=20)
    ax_absolutas.set_xlabel('')
    ax_absolutas.tick_params(axis='x', rotation=giro)

    if mostrar_valores:
        for p in ax_absolutas.patches:
            height = p.get_height()
            ax_absolutas.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                                  ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Frecuencias Relativas (solo si relativas=True)
    if relativas:
        # Tamaño de la serie
        serie_relativas = df[columna].value_counts(normalize=True)  # Acotamos el numero de barras si queremos

        
        if tamaño != False and type(tamaño) == int:
            serie_relativas = serie_relativas.head(tamaño)
            
        ax_relativas = axes[1]
        sns.barplot(x=serie_relativas.index, y=serie_relativas, ax=ax_relativas, palette='crest', hue=serie_relativas.index, legend=False)
        ax_relativas.set_ylabel('Frecuencia Relativa')
        ax_relativas.set_title(f'Distribución de {columna}',pad=20)
        ax_relativas.set_xlabel(columna)
        ax_relativas.tick_params(axis='x', rotation=giro)

        if mostrar_valores:
            for p in ax_relativas.patches:
                height = p.get_height()
                ax_relativas.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    plt.show()


def histo_box(df, col_num, k=1.5, bin=40, ajuste_y=None):
    """Visualiza un histograma con KDE y un boxplot para una columna numérica en un DataFrame.
    Parametros:
    - df: DataFrame, el conjunto de datos.
    - col_num: str, el nombre de la columna numérica.
    - k: float, el factor para calcular los límites del bigote (default: 1.5).
    - bin: int, el número de bins en el histograma (default: 40).
    - ajuste_y: int, el número máximo de valores en el eje y del histograma (default: None).
    """
    # Asegurar que max_y_values sea un valor numérico entero
    if ajuste_y is not None and not isinstance(ajuste_y, int):
        raise ValueError("max_y_values debe ser un valor numérico entero.")
    
    # Crear la figura y los ejes
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})

    # Graficar histograma y KDE en el primer eje
    sns.histplot(df[col_num], bins=bin, kde=True, ax=ax_hist, color="mediumslateblue", edgecolor="darkslateblue", alpha=0.5,
                  line_kws = {'linewidth':'2'},kde_kws={'bw_method': 0.5}).lines[0].set_color("darkslateblue")
    ax_hist.set_xlabel(col_num)
    ax_hist.set_title(f'Histograma y KDE ({col_num})',pad=20)
    ax_hist.set_ylabel('Frecuencia')

    # Limitar el número máximo de valores en el eje y del histograma si se especifica
    if ajuste_y is not None:
        ax_hist.set_ylim(0, ajuste_y)

    # Graficar boxplot en el segundo eje
    whisker_props={"color":"darkslateblue",
                   "linewidth": 1.5}
    box_props={"edgecolor":"darkslateblue",
               "linewidth": 1.5,
               "facecolor": "mediumslateblue", "alpha":0.5}
    median_props={"color":"darkslateblue",
                  "linewidth": 1.5}
    cap_props={"color": "darkslateblue",
               "linewidth": 1.5}
    sns.boxplot(x=df[col_num], ax=ax_box, whis=k, whiskerprops=whisker_props, boxprops=box_props, medianprops=median_props, capprops=cap_props)
    ax_box.set_title(f'Boxplot ({col_num})',pad=20)
    ax_box.set_xlabel(col_num)

    # Ajustar el diseño para que no haya superposiciones
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def outliers(df,col_num,k=1.5):
    """
    Identifica y cuenta los valores atípicos (outliers) en una columna numérica de un DataFrame.

    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - col_num (str): Nombre de la columna numérica en la que se buscarán outliers.
    - k (float, opcional): Factor de ampliación del rango intercuartílico para definir los límites. 
      Por defecto, se utiliza k=1.5, que es comúnmente usado.

    Imprime en consola:
    - Límite superior e inferior para detectar outliers.
    - Número de datos por encima del límite superior.
    - Número de datos por debajo del límite inferior.
    - Número total de datos fuera de los límites.

    Ejemplo de uso:
    >>> outliers(dataframe, 'columna_numerica', k=2.0)
    
    Nota: Esta función utiliza el método del rango intercuartílico (IQR) para identificar outliers.

    """
    Q1 = np.percentile(df[col_num], 25)
    Q3 = np.percentile(df[col_num], 75)
    IQR = Q3 - Q1
    lim_sup = Q3 + k * IQR
    lim_inf = Q1 - k * IQR

    # Recuento
    outliers_inf = df.loc[df[col_num]<lim_inf,[col_num]].value_counts().sum()
    outliers_sup = df.loc[df[col_num]>lim_sup,[col_num]].value_counts().sum()
    num_outliers = outliers_inf+outliers_sup

    # Display the number of outliers
    print(f'Lim. Superior: {lim_sup.round(2)}, Lim. Inferior: {lim_inf.round(2)}.\n'
          f'Numero de datos por encima: {outliers_sup}, numero de datos por debajo: {outliers_inf}.\n'
          f'Numero de datos fuera de límites: {num_outliers}')


def remove_outliers(df, col, k=1.5):
    """Elimina los outliers que esten fuera del limite"""
    # Calcular el rango intercuartílico (IQR)
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1

    # Definir límites para identificar outliers
    lower_limit = Q1 - k * IQR
    upper_limit = Q3 + k * IQR

    # Filtrar los valores que están fuera de los límites
    df_filtered = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

    return df_filtered


def plot_cat_num(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    """
    Crea gráficos de barras para visualizar la relación entre una variable categórica y una variable numérica.

    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - categorical_col (str): Nombre de la columna categórica en el eje x.
    - numerical_col (str): Nombre de la columna numérica en el eje y.
    - show_values (bool, opcional): Indica si mostrar los valores en las barras. Por defecto, es False.
    - measure (str, opcional): La medida de tendencia central a utilizar ('mean', 'median' o 'sum'). Por defecto, es 'mean'.

    Ejemplo de uso:
    >>> plot_cat_num(dataframe, 'columna_categorica', 'columna_numerica', show_values=True, measure='mean')

    Nota:
    - Si hay más de 5 categorías, los datos se dividen en grupos de 5 para facilitar la visualización.
    - Utiliza seaborn para crear gráficos de barras.

    """
    # Compatibilidad con Matplotlib y Seaborn
    sns.set()
    
    # Manejo de valores nulos
    if df[categorical_col].isnull().any() or df[numerical_col].isnull().any():
        raise ValueError("La función no admite valores nulos en las columnas categóricas o numéricas.")

    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    elif measure == "sum":
        grouped_data = df.groupby(categorical_col)[numerical_col].sum()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset].reset_index()

            # Crea el gráfico
            plt.figure(figsize=(20, 10))
            ax = sns.barplot(x=data_subset[categorical_col], y=data_subset[numerical_col], palette="magma")

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')
            # Muestra el gráfico
        plt.show()
                    
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='b'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=15, linewidths=20,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def get_features_cat_regression(df,target_col,pvalue=0.05,umbral_card=0.5):
    """
    Descripción:
        La función identifica las variables categóricas de un dataframe que se consideran features de una variable target en función de su correlación.

    Argumentos:
        df (Pandas Dataframe) : El DataFrame sobre el que trabajar.
        target_col: varible objetivo (numérica continua o discreta) del df.
        pvalue: valor que restado a 1 nos indica el intervalo de confianza para la identificación de features (cómo correlan con la vasriable target) 
        umbral_card: umbral de cardinalidad.

    Returns:
        cat_features: lista de las variables categóricas que han sido identificadas como features.
    """
    umbral_card=0.5
    if target_col not in df.columns:
        print(f"Error: la columna {target_col} no existe.")
        return None
    
    if not np.issubdtype(df[target_col].dtype, np.number) or (not any(df[col].nunique() / len(df) * 100 > umbral_card for col in df.columns)):
        print(f"Error: la columna {target_col} no es numérica y/o su cardinalidad es inferior a {umbral_card}.")
        return None
    
    if not isinstance(pvalue, float):
        print(f"Error: la variable {pvalue} no es float.")
        return None
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    cat_features = []
    for col in categorical_cols: # Calcular la correlación y p-values para cada columna numérica con 'target_col'
        contingency_table = pd.crosstab(df[col], df[target_col])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < pvalue:
            cat_features.append(col)
    return(cat_features)

# Plot Features Cat Regression
def plot_features_cat_regression(df,target_col="",columns=[],umbral_card=0.5,pvalue=0.05,with_indivudual_plot=False,umbral_corr=0):
    """
    Descripción:
        La función dibuja los histogramas de la variable objetivo para cada una de las features.

    Argumentos:
        df: El DataFrame sobre el que trabajar.
        target_col: varible objetivo (numérica continua o discreta) del df.
        columns: listado de variables categóricas. Por defecto está vacío.
        pvalue: valor que restado a 1 nos indica el intervalo de confianza para la identificación de features (cómo correlan con la variable target). Por defecto 0.05.
        with_individual_plot: argumento para dibujar el histograma individual o agrupado (por defecto).

    Returns:
        figure: histogramas
    """
    # Comprobación de los valores de entrada:
    if target_col not in df.columns:
        print(f"Error: la columna {target_col} no existe.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    if (df[target_col].nunique()/(len(df[target_col])*100))<umbral_card:
        print(f"Error: la cardinalidad de la columna {target_col} es inferior a {umbral_card}.")
        return None
    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue debe ser None o un número entre 0 y 1.")
        return None
    
    
    # Revisión del listado columns y creación de los gráficos.
    # Si columns está vacía:
    if columns==[]:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        num_features = []  # Filtra las columnas basadas en la correlación y el valor p
        for col in columns:
            if col != target_col:
                correlation, p_value = pearsonr(df[col], df[target_col])
                if abs(correlation) > umbral_corr and (pvalue is None or p_value < pvalue):
                    num_features.append(col)
        num_cols = len(num_features)
        fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5))
        for i, col in enumerate(num_features):
            ax = axes[i]
            sns.histplot(data=df, x=target_col, y=col, ax=ax, bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f'{col} vs {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()   
    # Si columns contiene info:
    else:
        columns_in_df=[]
        for col in columns:
            if col in df.columns:
                columns_in_df.append(col)
        if columns_in_df==[]:
            print(f"Error: las columnas no coinciden con las del df.")
            return None                
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        cat_features = []
        for col in categorical_cols: 
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_value, dof, expected= chi2_contingency(contingency_table)
            if p_value < pvalue:
                cat_features.append(col)
        if not cat_features:
            print("No se encontraron características categóricas significativas.")
            return None
        for col in columns_in_df:
            if col in cat_features:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=target_col, hue=col, palette='Set1', multiple='stack')
                plt.title(f'Histograma de {col} en relación con {target_col}')
                plt.xlabel(target_col)
                plt.ylabel('Frecuencia')
                plt.legend(title=col)
                plt.show()