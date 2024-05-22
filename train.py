from preprocess import *
from sklearn.ensemble import GradientBoostingRegressor
# import pandas as pd
# import numpy as np
# from datetime import datetime


class TrainModel(object):
    def __init__(self):
        pass

    @staticmethod
    def preprocess_dual_stage_pairwise(df, default_params):
        """Preprocess part of pairwise model

        Parameters
        ----------
        df : pandas DataFrame
            Raw Data
        default_params : dict
            Coffee params (refer to default_params in Coffee)

        Returns
        -------
        dict
            A dict with feature/label/model configs

        """

        backtrace_days = default_params["backtrace_days"]
        ratio = default_params["ratio"]

        date, feature, label = feature_transform(df, default_params)

        if default_params["backtrace_type"] == "similarity":
            (
                pairwise_feature,
                pairwise_label,
                pairwise_dates,
                pairwise_value,
                pairwise_similarity,
            ) = pairwise_transform_similarity(date, feature, label, default_params)
        else:
            # default to "previous" method
            (
                pairwise_feature,
                pairwise_label,
                pairwise_dates,
                pairwise_value,
            ) = pairwise_transform(date, feature, label, default_params)
            pairwise_similarity = None

        return {
            "features": pairwise_feature,
            "labels": pairwise_label,
            "dates": pairwise_dates,
            "similarity": pairwise_similarity,
            "original_value": pairwise_value,
            "backtrace_days": backtrace_days,
            "ratio": ratio,
        }

    @staticmethod
    def train_model_dual_stage_pairwise(trained_features, model_params=None):
        """Train part of pairwise model

        Parameters
        ----------
        trained_features : dict
            A dict like {'features': pairwise_feature,
                    'labels': pairwise_label}
        model_params : dict
            Params for GradientBoostingRepressor

        Returns
        -------
        model : obj
            Trained model
        """

        backtrace_days = trained_features["backtrace_days"]
        pairwise_feature = np.array(trained_features["features"])
        pairwise_label = np.array(trained_features["labels"])

        pairwise_feature = pairwise_feature[backtrace_days:]
        pairwise_label = pairwise_label[backtrace_days:]

        if model_params is not None:
            model = GradientBoostingRegressor(**model_params)
        else:
            model = GradientBoostingRegressor()
        model.fit(pairwise_feature, pairwise_label)

        return model

    @staticmethod
    def train_model_other_type(feature, label):
        # to add new models here
        pass


class PredictModel(object):
    def __init__(self, model, feature):
        self.model = model
        self.feature = feature

    def predict_model_dual_stage_pairwise(self):
        """Predict part of pairwise model

        Returns
        -------
        result : pandas DataFrame
            Forecast result

        """

        model = self.model
        pairwise_feature = np.array(self.feature["features"])
        pairwise_dates = self.feature["dates"]
        pairwise_similarity = self.feature["similarity"]
        predict_date = self.feature["predict_date"]
        ratio = self.feature["ratio"]
        pairwise_value = np.array(self.feature["original_value"])

        temp = pd.DataFrame(pairwise_dates, columns=["left_date", "right_date"])
        temp["similarity"] = pairwise_similarity
        temp["predict_value"] = model.predict(pairwise_feature)
        final_df = pd.concat(
            [temp, pd.DataFrame(pairwise_value, columns=["left_value", "right_value"])],
            axis=1,
        )
        final_df["candidate_value"] = final_df["left_value"] + final_df["predict_value"]

        result = []

        for i in predict_date:
            candidates = final_df[final_df.right_date == i][
                ["candidate_value", "similarity"]
            ]
            actual = final_df[final_df.right_date == i].right_value.mean()
            if pairwise_similarity is not None:
                length = int(len(candidates) * ratio)
                candidates = candidates.sort_values("similarity", ascending=False)
                candidates = candidates[:length]
                predict = sum(candidates.candidate_value) / len(candidates)
            else:
                length = int(len(candidates) * ratio)
                candidates = candidates.sort_values("candidate_value", ascending=False)
                candidates = candidates[length:-length]
                predict = sum(candidates.candidate_value) / len(candidates)
            result.append([i, np.expm1(actual), np.expm1(predict)])
        result = pd.DataFrame(result, columns=["date", "value", "predict"])

        return result
