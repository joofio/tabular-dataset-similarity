import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    ndcg_score,
    cohen_kappa_score,
)
import rbo
from sklearn.inspection import permutation_importance

import shap
import itertools
import scipy.stats as st
import random
from textdistance import (
    levenshtein,
    damerau_levenshtein,
    jaro_winkler,
    hamming,
)
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# for comparasion
def get_several_dif_dataset(
    data1,
    data2,
    categorical_cols,
    int_cols,
    cv,
    models=[DecisionTreeClassifier, LinearRegression],
):
    """
    This is the gold standard as of now. It is a function that takes in two datasets and
    returns the scores for each of the metrics.
    """

    le = preprocessing.OrdinalEncoder()
    le.fit(data1[categorical_cols].astype(str))
    data1[categorical_cols] = le.transform(data1[categorical_cols].astype(str))
    # le = preprocessing.OrdinalEncoder()
    # le.fit(data2[categorical_cols].astype(str))
    data2[categorical_cols] = le.transform(data2[categorical_cols].astype(str))

    r_cols = data1.columns
    result = {}
    for i in range(0, len(r_cols)):
        model = (
            models[0](random_state=42) if r_cols[i] in categorical_cols else models[1]()
        )
        metric = accuracy_score if r_cols[i] in categorical_cols else mean_squared_error
        X1 = data1.drop(r_cols[i], axis=1)
        y1 = data1[r_cols[i]]
        X2 = data2.drop(r_cols[i], axis=1)
        y2 = data2[r_cols[i]]

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X1, y1, test_size=0.2, random_state=42
        )
        #  X_train2, X_test2, y_train2, y_test2 = train_test_split(
        #      X2, y2, test_size=0.2, random_state=42
        #  )
        model.fit(X_train1, y_train1)
        real_real = metric(y_test1, model.predict(X_test1))
        real_synth = metric(y2, model.predict(X2))
        # =cross_val_score(lr, X_train1, y_train1, cv=cv, scoring=metric)
        # =cross_val_score(lr, X_train1, y2, cv=cv, scoring=metric)

        result[r_cols[i]] = real_synth / real_real
    return result


def aggregate_data_cross(
    real_data, synth_data, categorical_values, continuous_values, cv
):

    """
    ???
    """
    real_synth_dif = get_several_dif_dataset(
        real_data, synth_data, categorical_values, continuous_values, cv
    )
    print(real_synth_dif)
    synth_real_dif = get_several_dif_dataset(
        synth_data, real_data, categorical_values, continuous_values, cv
    )
    print(synth_real_dif)
    synth_real_score = {k: np.mean(v) for k, v in synth_real_dif.items()}
    real_synth_score = {k: np.mean(v) for k, v in real_synth_dif.items()}
    # synth_real_score_df=pd.DataFrame.from_dict(synth_real_score,orient='index',columns=["Metric"])
    # real_synth_score_df=pd.DataFrame.from_dict(real_synth_score,orient='index',columns=["Metric"])
    final_score = {}
    for k, v in synth_real_score.items():
        # print(synth_real_score[k],real_synth_score[k])
        final_score = synth_real_score[k] / real_synth_score[k]
    return final_score


def check_variance(dict_):
    var_dict = {}
    for key, value in dict_.items():
        var_dict[key] = {}
        for k2, v2 in dict_[key].items():
            #  print(key,k2)
            var_dict[key][k2] = np.var(v2)
    return var_dict


def get_several_feat_imp_dataset_2(
    data,
    categorical_cols,
    int_cols,
    rep=5,
    seed=42,
    test_size=0.05,
    models=[DecisionTreeClassifier, LinearRegression],
):
    """

    1. por cada coluna
    2. por cada nr de repitições
    3. treinar modelo
    4. ir buscar feature importance
    5. fazer a media das medias

    result:{Predicted:{feature1:[v_rep1,v_rep2,v_rep3],feature2:[v_rep1,v_rep2,v_rep3]}}


    """

    r_cols = data.columns
    result = {}
    #  print(result)
    np.random.seed(seed)
    random.seed(seed)
    for i in range(0, len(r_cols)):
        # print("testing...", r_cols[i])
        l_feats = {k: [] for k in r_cols if k != r_cols[i]}
        for r in range(0, rep):
            #     print("rep",r)
            n = random.randint(0, 100)
            if r_cols[i] in categorical_cols:
                if hasattr(models[0], "random_state"):
                    model = models[0].set_params(random_state=np.random.randint(1, 20))
                else:
                    model = models[0]
            else:
                if hasattr(models[1], "random_state"):
                    model = models[1].set_params(random_state=np.random.randint(1, 20))

                else:
                    model = models[1]
            # metric = (
            #    "roc_auc_score"
            #    if r_cols[i] in categorical_cols
            #    else "neg_mean_absolute_error"
            # )
            X = data.drop(r_cols[i], axis=1)
            y = data[r_cols[i]]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=n
            )  # just for bootstrap
            # print(X_train)
            t = model.fit(X_train, y_train)

            if hasattr(model, "feature_importances_"):
                # print(r)
                # print(t.feature_names_in_)
                # print(t.feature_importances_)
                # feats = {}
                for g in zip(t.feature_names_in_, t.feature_importances_):
                    # print(g)
                    l_feats[g[0]].append(g[1])
            #        print(l_feats)
            else:
                r = permutation_importance(
                    t, X_train, y_train, n_repeats=30, random_state=n
                )

                for g in zip(X_train.columns, r.importances_mean):
                    l_feats[g[0]].append(g[1])

            result[r_cols[i]] = l_feats
    return result


def create_scores_v2(result1, result2):
    """
    does not work for more than two datasets
    #https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
    #https://stats.stackexchange.com/questions/51295/comparison-of-ranked-lists
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html


    """
    keys = result1.keys()
    scores_ = {}
    for target in keys:
        # print(col)
        ftkeys = [key for key in keys if key != target]

        #  print(result1[col])
        m1 = {k: np.mean(v) for k, v in result1[target].items()}
        m2 = {k: np.mean(v) for k, v in result2[target].items()}
        # print(m1)
        # print(m2)
        x1_rank = st.rankdata(
            [-1 * el for el in m1.values()], method="ordinal"
        )  # avoid tie
        x1_rank_dict = {k: v for k, v in zip(m1.keys(), x1_rank)}
        # print(x1_rank_dict)

        x2_rank = st.rankdata(
            [
                -1 * el if el != 0 else el * np.random.randint(1, 10) * 0.00001 * -1
                for el in m2.values()
            ],
            method="ordinal",  # avoid tie
        )  # avoid being zero
        x2_rank_dict = {k: v for k, v in zip(m2.keys(), x2_rank)}
        #    print(x2_rank)
        # print(x2_rank_dict)
        # print(m2)

        true_score = []
        model_score = []
        true_score_rank = []
        model_score_rank = []
        for key in ftkeys:
            true_score_rank.append(x1_rank_dict[key])
            model_score_rank.append(x2_rank_dict[key])
            true_score.append(m1[key])
            model_score.append(m2[key])

        #  print(true_score)
        #  print(model_score)
        #  print(true_score_rank)
        #  print(model_score_rank)

        true_score_rank_join = "".join(str(int(e)) for e in true_score_rank)
        model_score_rank_join = "".join(str(int(e)) for e in model_score_rank)

        #  l_=ndcg_score([true_score_rank],[model_score])
        n_l = ndcg_score([true_score_rank], [model_score_rank])
        #
        def mae_over_max(mae, max_):
            if max_ == 0:
                return 1
            else:
                return mae / max_

        sc = {}
        sc["ndgc_score"] = n_l
        sc["cohen_kappa_score"] = cohen_kappa_score(true_score_rank, model_score_rank)

        sc["r2_score"] = r2_score(true_score, model_score)
        sc["levenshtein_normalized_similarity"] = levenshtein.normalized_similarity(
            true_score_rank, model_score_rank
        )
        sc["kendalltau"] = st.kendalltau(true_score_rank, model_score_rank)[0]
        sc["weightedtau"] = st.weightedtau(true_score_rank, model_score_rank)[0]
        sc["rbo"] = rbo.RankingSimilarity(true_score_rank, model_score_rank).rbo()

        sc[
            "damerau_levenshtein_normalized_similarity"
        ] = damerau_levenshtein.normalized_similarity(true_score_rank, model_score_rank)
        sc["jaro_winkler_normalized_similarity"] = jaro_winkler.normalized_similarity(
            true_score_rank, model_score_rank
        )

        sc["hamming_normalized_similarity"] = hamming.normalized_similarity(
            true_score_rank, model_score_rank
        )

        scores_[target] = {
            "results": sc,
            "true_score": true_score,
            "model_score": model_score,
            "true_score_rank": true_score_rank,
            "model_score_rank": model_score_rank,
            "true_score_rank_join": true_score_rank_join,
            "model_score_rank_join": model_score_rank_join,
        }
    # for aggregated scores:
    full_df = None
    for k, v in scores_.items():
        # print(scores_[k]["results"])
        res_df = pd.DataFrame(scores_[k]["results"], index=[0])
        if full_df is None:
            full_df = res_df
        else:
            full_df = pd.concat([full_df, res_df])
        # res_df.loc['mean_column'] = res_df.mean()
    # full_df["mean_row"] = full_df.mean(numeric_only=True, axis=1)
    # print(full_df)
    full_df.loc["mean"] = full_df.mean()

    scores_["aggregated"] = full_df.loc["mean"].to_dict()
    return scores_


def test_two_datasets(
    data, data_1, categorical_values, continuous_values, reps=10, seed=42
):

    result_1 = get_several_feat_imp_dataset_2(
        data, categorical_values, continuous_values, reps, seed=seed
    )
    result_2 = get_several_feat_imp_dataset_2(
        data_1, categorical_values, continuous_values, reps, seed=seed
    )
    sc = create_scores_v2(result_1, result_2)

    sc["cross"] = aggregate_data_cross(
        data_1, data, categorical_values, continuous_values, 10
    )

    return sc


def plot_plotly(
    plot_data,
    cols=[
        "cross",
        "ndgc_score",
        "r2_score",
        "kendalltau",
        "weightedtau",
        "rbo",
        "levenshtein_normalized_similarity",
    ],
):
    df_plot = pd.DataFrame.from_dict(plot_data)
    xx = pd.melt(df_plot, value_vars=df_plot.columns, ignore_index=False)
    # print(xx)
    xx = xx.reset_index()

    fig = px.line(xx[xx["index"].isin(cols)], x="variable", y="value", color="index")
    fig.show()
    fig.write_image("Viz/difference.png", width=1000, height=600)
    return xx[xx["index"].isin(cols)]


def trial_permutatin(
    data,
    categorical_values,
    continuous_values,
    cv,
    reps=20,
    nr_cols_to_test=7,
    models=[DecisionTreeClassifier, LinearRegression],
):

    plot_data = {}
    local_plot_data = {}
    for i in range(0, nr_cols_to_test + 1):  # nr of columns
        plot_data["run " + str(i)] = {"cross": []}
        print("run nr {}".format(i), "++" * 40)
        local_plot_data = {"cross": []}

        for j in range(0, reps):  # nr of repetitions

            print("reps", str(j + 1))
            random.seed(j)
            data_1 = data.copy()
            # for k in range(0, i):
            #    print("k",k)
            if i > 0:
                cols_to_shuffle = random.sample(range(0, len(data.columns)), i)
                print(cols_to_shuffle)
                print(data.columns[cols_to_shuffle])

                #  print(list(range(0, i)))
                data_1.iloc[:, cols_to_shuffle] = np.random.permutation(
                    data_1.iloc[:, cols_to_shuffle].values
                )
            # print(data_1)
            seed = np.random.randint(1, 20)

            result_1 = get_several_feat_imp_dataset_2(
                data,
                categorical_values,
                continuous_values,
                reps,
                seed=seed,
                models=models,
            )
            result_2 = get_several_feat_imp_dataset_2(
                data_1,
                categorical_values,
                continuous_values,
                reps,
                seed=seed,
                models=models,
            )
            sc = create_scores_v2(result_1, result_2)
            # print(sc["aggregated"])
            for k, v in sc["aggregated"].items():
                if local_plot_data.get(k) is None:
                    local_plot_data[k] = [v]
                else:
                    local_plot_data[k].append(v)

            # print(local_plot_data)
            local_plot_data["cross"].append(
                aggregate_data_cross(
                    data_1, data, categorical_values, continuous_values, cv
                )
            )
        plot_data["run " + str(i)] = {k: np.mean(v) for k, v in local_plot_data.items()}
        plot_data["run " + str(i)]["debug"] = local_plot_data

    return plot_data


#### OLD ############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


def overall_ratio_new(data, categorical_values, continuous_values, rep, method="shap"):

    """
    nao faz sentido combinação de relações ???
    """
    results = {}
    scores_ = {}
    dfs_ = {}
    np.random.seed(42)
    for idx, dataset in enumerate(data):
        seed = np.random.randint(1, 20)
        # print(seed)
        if method == "shap":
            results[idx] = get_several_feat_imp_dataset_shap(
                dataset, categorical_values, continuous_values, rep, seed=seed
            )
        elif method == "feat_imp":
            results[idx] = get_several_feat_imp_dataset_2(
                dataset, categorical_values, continuous_values, rep, seed=seed
            )
        else:
            raise Exception("no known method")

    sc = create_scores_v2(results[0], results[1])
    for s in sc[data[0].columns[0]].keys():
        if s in scores_.keys():
            scores_[s].append({k: v[s] for k, v in sc.items()})
        else:
            scores_[s] = [({k: v[s] for k, v in sc.items()})]
    for k, v in scores_.items():

        res_df = pd.DataFrame(scores_[k])
        # res_df.loc['mean_column'] = res_df.mean()
        res_df["mean_row"] = res_df.mean(numeric_only=True, axis=1)
        dfs_[k] = res_df
    return dfs_


def overall_ratio(data, categorical_values, continuous_values, cv, rep, method="shap"):
    results = {}
    scores_ = {}
    dfs_ = {}
    np.random.seed(42)
    indexes_comb = []
    for idx, dataset in enumerate(data):
        seed = np.random.randint(1, 20)
        if method == "shap":
            results[idx] = get_several_feat_imp_dataset_shap(
                dataset, categorical_values, continuous_values, cv, rep, seed=seed
            )
        elif method == "feat_imp":
            results[idx] = get_several_feat_imp_dataset_2(
                dataset, categorical_values, continuous_values, cv, rep, seed=seed
            )
        else:
            raise Exception("no known method")

    #  print(results)
    for idx_2, comb in enumerate(itertools.product(range(0, len(data)), repeat=2)):
        if idx_2 in [1, 2]:
            indexes_comb.append(str(comb))
            # print(comb)
            # print("--"*40)
            sc = create_scores_v2(results[comb[0]], results[comb[1]])
            for s in sc[data[0].columns[0]].keys():
                if s in scores_.keys():
                    scores_[s].append({k: v[s] for k, v in sc.items()})
                else:
                    scores_[s] = [({k: v[s] for k, v in sc.items()})]
    for k, v in scores_.items():

        res_df = pd.DataFrame(scores_[k], index=indexes_comb)
        res_df.loc["mean_column"] = res_df.mean()
        res_df["mean_row"] = res_df.mean(numeric_only=True, axis=1)
        dfs_[k] = res_df
    return dfs_


def get_several_feat_imp_dataset_shap(
    data, categorical_cols, int_cols, cv, rep=5, seed=42
):
    """ """

    r_cols = data.columns
    result = {}
    #  print(result)
    np.random.seed(seed)
    random.seed(seed)
    for i in range(0, len(r_cols)):  # for all columns as target
        # print("target: ", r_cols[i])

        l_feats = {k: [] for k in r_cols if k != r_cols[i]}
        for r in range(0, rep):  # for number of repetitions
            #   print("rep: ", r)
            n = random.randint(0, 100)
            model = (
                DecisionTreeClassifier(random_state=np.random.randint(1, 20))
                if r_cols[i] in categorical_cols
                else DecisionTreeRegressor(random_state=np.random.randint(1, 20))
            )
            # metric = (
            #     "roc_auc"
            #     if r_cols[i] in categorical_cols
            #     else "neg_mean_absolute_error"
            # )
            X = data.drop(r_cols[i], axis=1)
            y = data[r_cols[i]]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=n
            )  # for bootstrapping

            t = model.fit(X_train, y_train)
            explainer = shap.TreeExplainer(t)
            # compute SHAP values
            shap_values = explainer.shap_values(X_train)
            # print(metric)
            class_list_mean = []
            # print(shap_values)
            #   print(type(shap_values))
            if type(shap_values) != list:  # regression
                #     print(shap_values.shape)
                means = np.mean(shap_values, axis=0)
            #     print(means.shape)
            # means=list(shap_values).mean(axis=0)
            else:  # classification
                #     print(len(shap_values))
                for l in shap_values:
                    class_list_mean.append(l.mean(axis=0))
                # print(class_list_mean[0].shape)
                means = np.mean(class_list_mean, axis=0)
            # print(means)
            cols = list(X.columns)
            # print(cols)
            # print(means,cols)
            for g in zip(cols, means):

                l_feats[g[0]].append(g[1])
            # print(l_feats)
            result[r_cols[i]] = l_feats
    return result
