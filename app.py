import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Application Isothermes d'Adsorption", layout="wide")

# Menu latéral
menu = st.sidebar.radio("Navigation", ["\U0001F3E0 Présentation", "\U0001F9EA Analyse des données", "\U0001F52C Interprétation des résultats"])

# Présentation
if menu == "\U0001F3E0 Présentation":
    st.title("\U0001F331 Comparaison des Modèles d'Isothermes d'Adsorption")

    st.markdown("""
    **Optimisez vos expériences avec les meilleurs modèles d’adsorption.**

    Cette application web vous permet de **comparer différents modèles d’isothermes d’adsorption**. Que vous soyez chercheur, ingénieur ou étudiant, cet outil vous aide à choisir le meilleur modèle pour vos données expérimentales d'adsorption. 

    ---
    ### \U0001F4D8 Modèles disponibles
    **Modèles à deux paramètres**
    - Langmuir
    - Freundlich
    - Tempkin
    - Jovanovich
    - Dubinin-Radushkevich
    - Modèle personnalisé *(modèle innovant)*

    **Modèles à trois paramètres**
    - Sips
    - Toth
    - Khan
    - Redlich-Peterson

    ---
    ### ⚙️ Fonctionnalités
    1. **Importation de données CSV** contenant les colonnes `Ce` et `qe`
    2. **Ajustement automatique des modèles sélectionnés**
    3. **Affichage des courbes d’ajustement**
    4. **Comparaison des performances selon R², AIC, AICc, BIC, RMSE, MAE**
    5. **Recommandation du meilleur modèle selon AIC et BIC**
    6. **Export des résultats au format CSV**

    ---
    ### 🚀 Pourquoi utiliser notre application ?
    - Interface simple et intuitive
    - Comparaison visuelle et statistique claire
    - Analyse complète avec export des résultats possible
    - Intégration d’un **modèle personnalisé**, plus flexible dans certains cas expérimentaux

    ---
    ### \U0001F4E4 Télécharger les résultats
    Vous pouvez désormais **exporter vos résultats** (paramètres ajustés, courbes, statistiques) en **CSV** depuis la section d'analyse.

    ---
    ### \U0001F4E9 Contact et Support
    Pour toute question ou suggestion, veuillez contacter :
    - **Nom :** AGOSSOU Gildas Fiacre
    - **Email :** gildas.agossou950@gmail.com

    ---
    **© 2025 AGOSSOU Gildas Fiacre. Tous droits réservés.**
    """)

# Page d'analyse
elif menu == "\U0001F9EA Analyse des données":
    st.title("\U0001F9EA Analyse des Données d'Adsorption")
    st.markdown("""
    Importez vos données, sélectionnez les modèles et visualisez les courbes d'ajustement ainsi que les statistiques de performance.
    """)

    # Modèles d'adsorption
    def langmuir(Ce, q_m, K):
        return (q_m * K * Ce) / (1 + K * Ce)

    def freundlich(Ce, K, n):
        return K * Ce ** (1 / n)

    def tempkin(Ce, B, A):
        Ce = np.array(Ce)
        Ce = np.where(Ce <= 0, 1e-6, Ce)
        A = np.abs(A) + 1e-6
        return B * np.log(A * Ce)

    def dubinin_radushkevich(Ce, q_m, K):
        return q_m * np.exp(-K * (np.log(1 + 1 / Ce)) ** 2)

    def jovanovich(Ce, q_m, K):
        return q_m * (1 - np.exp(-K * Ce))

    def sips(Ce, q_m, K, n):
        return (q_m * K * Ce ** n) / (1 + K * Ce ** n)

    def toth(Ce, q_m, K, n):
        return (q_m * K * Ce) / ((1 + (K * Ce) ** n) ** (1 / n))

    def khan(Ce, q_m, K, n):
        return (q_m * K * Ce) / (1 + (K * Ce) ** n)

    def redlich_peterson(Ce, K, a, g):
        return (K * Ce) / (1 + a * Ce ** g)

    def custom_model(Ce, q_m, K):
        alpha = 0.45
        beta = 0.55
        return (q_m * K * Ce ** alpha) / (1 + K * Ce ** beta)

    model_dict_2_params = {
        "Langmuir": (langmuir, [405, 1.0]),
        "Freundlich": (freundlich, [1.0, 1.0]),
        "Tempkin": (tempkin, [1.0, 1.0]),
        "Dubinin-Radushkevich": (dubinin_radushkevich, [405, 1.0]),
        "Jovanovich": (jovanovich, [405, 1.0]),
        "Modèle personnalisé": (custom_model, [405, 1.0])
    }

    model_dict_3_params = {
        "Sips": (sips, [405, 1.0, 1.0]),
        "Toth": (toth, [405, 1.0, 1.0]),
        "Khan": (khan, [405, 1.0, 1.0]),
        "Redlich-Peterson": (redlich_peterson, [1.0, 1.0, 0.9])
    }

    uploaded_file = st.file_uploader("\U0001F4C4 Importez un fichier CSV contenant les colonnes 'Ce' et 'qe'", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        Ce = data["Ce"].values
        qe = data["qe"].values

        selected_models_2_params = st.multiselect("Sélectionnez les modèles à deux paramètres :", list(model_dict_2_params.keys()), default=list(model_dict_2_params.keys()))
        selected_models_3_params = st.multiselect("Sélectionnez les modèles à trois paramètres :", list(model_dict_3_params.keys()), default=list(model_dict_3_params.keys()))

        selected_models = selected_models_2_params + selected_models_3_params

        if selected_models:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Ce, y=qe, mode='markers', name='Données expérimentales', marker=dict(color='black')))

            results = []
            for name in selected_models:
                if name in model_dict_2_params:
                    model_func, initial_params = model_dict_2_params[name]
                else:
                    model_func, initial_params = model_dict_3_params[name]

                try:
                    params, _ = curve_fit(model_func, Ce, qe, p0=initial_params, maxfev=10000)
                    qe_pred = model_func(Ce, *params)
                    residuals = qe - qe_pred
                    sse = np.sum(residuals**2)
                    rmse = np.sqrt(np.mean(residuals**2))
                    mae = np.mean(np.abs(residuals))
                    r2 = r2_score(qe, qe_pred)

                    n_obs = len(qe)
                    k = len(params)
                    aic = n_obs * np.log(sse/n_obs) + 2 * k
                    aicc = aic + (2 * k * (k + 1)) / (n_obs - k - 1) if n_obs - k - 1 > 0 else np.nan
                    bic = n_obs * np.log(sse/n_obs) + k * np.log(n_obs)

                    results.append({"Modèle": name, "R²": r2, "AIC": aic, "AICc": aicc, "BIC": bic, "SSE": sse, "MAE": mae, "RMSE": rmse})

                    fig.add_trace(go.Scatter(x=Ce, y=qe_pred, mode='lines', name=name))
                except Exception as e:
                    st.error(f"❌ Erreur pour {name} : {e}")

            st.plotly_chart(fig, use_container_width=True)

            results_df = pd.DataFrame(results).sort_values(by=["AIC", "BIC"])
            st.dataframe(results_df.style.format({"R²": "{:.4f}", "AIC": "{:.2f}", "AICc": "{:.2f}", "BIC": "{:.2f}", "SSE": "{:.2f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}))

            best_model_aic = results_df.loc[results_df["AIC"].idxmin()]
            best_model_bic = results_df.loc[results_df["BIC"].idxmin()]

            st.markdown(f"### 🎯 Meilleur modèle selon AIC : **{best_model_aic['Modèle']}**")
            st.markdown(f"### 🎯 Meilleur modèle selon BIC : **{best_model_bic['Modèle']}**")

elif menu == "\U0001F52C Interprétation des résultats":
    st.title("\U0001F52C Interprétation des Résultats")

    st.markdown("""
    ### 🔎 Interprétation des Modèles d'Adsorption

    Cette section vous permet d'interpréter les résultats des modèles d'adsorption en fonction des paramètres ajustés et des valeurs de performance. Ci-dessous, vous trouverez l'explication des principaux modèles utilisés dans cette application, ainsi que leur signification dans le contexte des résultats expérimentaux.

    #### 1. **Langmuir (Modèle à deux paramètres)**

    L’isotherme d’adsorption de Langmuir permet de comprendre comment un soluté ou un gaz se fixe à la surface d’un adsorbant. Elle suppose essentiellement que les molécules de l’adsorbat forment une monocouche sur la surface de l’adsorbant, ce qui signifie que dès qu’un site de la surface est occupée par une molécule, aucune autre molécule ne peut venir s’y fixer. Ce modèle stipule aussi que les molécules de l’adsorbat n’interagissent pas entre elles.

    **Paramètres :**
    - \(q_m\) : Capacité maximale d’adsorption (mg/g) — La quantité maximale que l'adsorbant peut capturer.
    - \(K\) : Constante d’équilibre (L/mg) — Mesure l'affinité entre l'adsorbant et l'adsorbat.

    **Interprétation de \(R_L\) :**
    Le facteur de Langmuir, \(R_L\), donne une indication sur la favorabilité de l'adsorption. Il est calculé comme suit :

    \[
    R_L = \frac{1}{1 + K C_0}
    \]
    - Si \(R_L > 1\), l'adsorption est défavorable.
    - Si \(R_L = 1\), l'adsorption est linéaire.
    - Si \(0 < R_L < 1\), l'adsorption est favorable.
    - Si \(R_L = 0\), il y a une adsorption irréversible.

    #### 2. **Freundlich (Modèle à deux paramètres)**

    Le modèle de Freundlich est utilisé pour les systèmes d'adsorption hétérogènes, où les sites d'adsorption n'ont pas la même énergie. Ce modèle indique qu'il existe des sites d'adsorption avec des affinités différentes pour l'adsorbat.

    **Paramètres :**
    - \(K\) : Capacité d’adsorption relative (mg/g (L/g)ⁿ) — Plus \(K\) est élevé, plus l'adsorbant est efficace.
    - \(n\) : Indicateur de l'hétérogénéité des sites — Si \(n > 1\), l'adsorption est favorable ; sinon, elle est défavorable.

    #### 3. **Tempkin (Modèle à deux paramètres)**

    Le modèle de Tempkin décrit une adsorption qui dépend de l'enthalpie d'adsorption et du facteur de température. Il est souvent utilisé lorsque l'adsorption est exothermique et que les interactions entre les molécules d'adsorbat ne peuvent pas être ignorées.

    **Paramètres :**
    - \(A\) : Constante d'équilibre.
    - \(B\) : Paramètre lié à l'enthalpie d'adsorption.

    #### 4. **Dubinin-Radushkevich (Modèle à deux paramètres)**

    Ce modèle est utilisé pour caractériser des surfaces d'adsorption hétérogènes et permet de mieux comprendre les mécanismes d'adsorption de type chimique. Il est basé sur la notion de microporosité.

    **Paramètres :**
    - \(q_m\) : Capacité d’adsorption maximale.
    - \(K\) : Constante qui prend en compte l'énergie de l'adsorption.

    #### 5. **Jovanovich (Modèle à deux paramètres)**

    Le modèle de Jovanovich est un modèle empirique qui est souvent utilisé pour l'adsorption dans des systèmes complexes où les interactions entre les molécules sont importantes.

    **Paramètres :**
    - \(q_m\) : Capacité d'adsorption maximale.
    - \(K\) : Constante d'adsorption.

    #### 6. **Modèle personnalisé (Modèle à deux paramètres)**

    Le modèle personnalisé proposé est basé sur des exponents de type fractionnaire, qui permettent de modéliser des mécanismes d'adsorption complexes, souvent observés dans les systèmes industriels.

    **Paramètres :**
    - \(q_m\) : Capacité d'adsorption maximale.
    - \(K\) : Constante d'adsorption.

    #### 7. **Sips (Modèle à trois paramètres)**

    Ce modèle combine les aspects des modèles de Langmuir et de Freundlich. Il est utilisé pour des surfaces ayant une distribution d'énergie non uniforme.

    **Paramètres :**
    - \(q_m\) : Capacité maximale d'adsorption.
    - \(K\) : Constante d’équilibre.
    - \(n\) : Paramètre qui modifie l'influence de la concentration.

    #### 8. **Toth (Modèle à trois paramètres)**

    Le modèle de Toth est utilisé pour les systèmes hétérogènes où l'adsorption n'est pas linéaire, souvent en raison des effets de la température ou de la pression.

    **Paramètres :**
    - \(q_m\) : Capacité maximale d’adsorption.
    - \(K\) : Constante d’équilibre.
    - \(n\) : Paramètre d'exponentiation.

    #### 9. **Khan (Modèle à trois paramètres)**

    Le modèle de Khan est utilisé pour l'adsorption sur des surfaces non homogènes, souvent lorsque les sites d'adsorption sont influencés par des facteurs extérieurs.

    **Paramètres :**
    - \(q_m\) : Capacité maximale d'adsorption.
    - \(K\) : Constante d’équilibre.
    - \(n\) : Paramètre influençant la forme de l’isotherme.

    #### 10. **Redlich-Peterson (Modèle à trois paramètres)**

    Ce modèle est souvent utilisé dans les systèmes où les isothermes d'adsorption ne peuvent pas être modélisés par des modèles simples comme Langmuir ou Freundlich. Il est plus flexible pour s'adapter à une grande variété de systèmes.

    **Paramètres :**
    - \(K\) : Constante d’adsorption.
    - \(a\) : Paramètre influençant la forme de l'isotherme.
    - \(g\) : Paramètre lié à la concentration.

    ---
    """)



