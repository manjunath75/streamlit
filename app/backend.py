import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    """
    Trains models required for recommendation.
    Returns trained model or artifacts as dictionary.
    """
    if params is None:
        params = {}
    ratings_df = load_ratings()
    bow_df = load_bow()

    # =========================
    # 1. Clustering (KMeans)
    # =========================
    if model_name == models[2]:  # "Clustering"
        n_clusters = params.get("n_clusters", 5)

        X = bow_df.drop(columns=["doc_id", "doc_index"])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        return {
            "model": kmeans,
            "features": X.columns.tolist()
        }

    # =========================
    # 2. Clustering with PCA
    # =========================
    if model_name == models[3]:  # "Clustering with PCA"
        n_clusters = params.get("n_clusters", 5)
        n_components = params.get("n_components", 20)

        X = bow_df.drop(columns=["doc_id", "doc_index"])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_pca)

        return {
            "pca": pca,
            "model": kmeans
        }

    # =========================
    # 3. KNN Collaborative Filtering
    # =========================
    if model_name == models[4]:  # "KNN"
        user_item = ratings_df.pivot_table(
            index="user", columns="item", values="rating"
        ).fillna(0)

        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(user_item)

        return {
            "model": knn,
            "matrix": user_item
        }

    # =========================
    # 4. NMF (Matrix Factorization)
    # =========================
    if model_name == models[5]:  # "NMF"
        n_components = params.get("n_components", 15)

        user_item = ratings_df.pivot_table(
            index="user", columns="item", values="rating"
        ).fillna(0)

        nmf = NMF(n_components=n_components, random_state=42)
        nmf.fit(user_item)

        return {
            "model": nmf,
            "user_item_matrix": user_item
        }

    # =========================
    # 5. Neural Network (Simple Embedding Proxy)
    # =========================
    if model_name == models[6]:  # "Neural Network"
        # Simple fallback: matrix factorization style
        n_components = params.get("n_components", 20)

        user_item = ratings_df.pivot_table(
            index="user", columns="item", values="rating"
        ).fillna(0)

        nmf = NMF(n_components=n_components, random_state=42)
        user_embeddings = nmf.fit_transform(user_item)

        return {
            "user_embeddings": user_embeddings,
            "item_embeddings": nmf.components_
        }

    # =========================
    # 6. Regression with Embeddings
    # =========================
    if model_name == models[7]:
        X = ratings_df[["user", "item"]]
        y = ratings_df["rating"]

        model = LinearRegression()
        model.fit(X, y)

        return {"model": model}

    # =========================
    # 7. Classification with Embeddings
    # =========================
    if model_name == models[8]:
        X = ratings_df[["user", "item"]]
        y = (ratings_df["rating"] >= 4).astype(int)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        return {"model": model}

    return None



# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
