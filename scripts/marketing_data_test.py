import json
from datetime import datetime

import sklearn.metrics
import argparse
from econml.dml import SparseLinearDML, CausalForestDML
from econml.validate import DRTester
import collinearity
from itertools import product
import joblib
import numpy as np
import os
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
import sys
import logging
from sklearn.model_selection import KFold

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def log_transform(x: pd.DataFrame, skew_tresh: int = 3) -> pd.DataFrame:
    """
    Logic to apply as function to make the log scaling. Scaling to log (x+1) whenever conditions
    are met:
        1. More than 2 unique values - some variables are binary, which won't have a skew
        2. All values >= 0, so log scaling can be  usd
        3. Skew is above a threshold
    :param x: Input data frame
    :param skew_tresh:
    :return: Scaled data frame
    """
    return pd.DataFrame(
        {
            c: np.log(y + 1) if (len(np.unique(y)) > 2 and np.all(y >= 0) and scipy.stats.skew(y) > skew_tresh) else y
            for c, y in x.items()
        }
    )


def scale_marketing_data(X: pd.DataFrame, T: pd.DataFrame, log_scale_skew_thresh: float | int) -> pd.DataFrame:
    """
    Apply logarithmic scaling for data, assumed to be highly skewed marketing data nuisance
    features. Treatment features are assumed to need only standard scaling

    :param X: Nuisance variables
    :param T: Treatment Variables
    :param log_scale_skew_thresh: Skew threshold for logarithmic scaling of each variable
    :return: Tuple of scaled nuisance and traetment varibles
    """

    # TODO: Update to use log scaling too. Why not?
    tscale = StandardScaler()
    T = tscale.fit_transform(T)

    X_unscaled = pd.DataFrame(X)
    xscale1 = StandardScaler()
    transform_fun = lambda x: log_transform(x, skew_tresh=log_scale_skew_thresh)
    transformer = FunctionTransformer(transform_fun)
    x_pipe = Pipeline(steps=[('transformer', transformer), ('scaler', xscale1)])
    X = pd.DataFrame(x_pipe.fit_transform(X), columns=X_unscaled.columns.values)

    scaled_stats = pd.DataFrame(X.describe()).transpose()
    scaled_stats['skew'] = X.skew()
    orig_stats = pd.DataFrame(X_unscaled.describe()).transpose()
    orig_stats['skew'] = X_unscaled.skew()

    logger.info("Used combined log standard scaling")
    full_stats = pd.concat([scaled_stats.add_suffix('_scale'), orig_stats.add_suffix('_orig')], axis=1)
    print(full_stats.to_string())

    return X, T


def remove_collinear_features(X: pd.DataFrame, y: np.array, T: pd.DataFrame, collinearity_thresh: float):
    """
    Remove collinear features from both Nuisance and Treatment variables. Just a wrapper around
    SelectNonCollinear - no custom logic.

    :param X:  Nuisance
    :param y: Outcome
    :param T:  Treatment
    :param collinearity_thresh: Threshold to pass to SelectNonCollinear
    :return:
    """
    logger.info(f"Starting collinearity thresh {collinearity_thresh} selection on X")
    selector = collinearity.SelectNonCollinear(collinearity_thresh)
    selector.fit(X.to_numpy(), y)
    mask = selector.get_support()
    X = X.loc[:, mask]
    logger.info(f"X now {X.shape}")

    logger.info(f"Starting collinearity thresh {collinearity_thresh} selection on T")
    selector = collinearity.SelectNonCollinear(collinearity_thresh)
    selector.fit(T, y)
    mask = selector.get_support()
    T = T[:, mask]
    logger.info(f"T now {T.shape}")

    return X, T


class XGBClassAUCScore(XGBClassifier):
    """
    Wrapper around an XGBoost classifier that returns the roc_auc as the score
    """

    def __init__(self, n_estimators=500, max_depth=9, learning_rate=0.05, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, **kwargs)

    def score(self, X, y=None):
        y_pred = self.predict_proba(X)
        return roc_auc_score(y, y_pred[:, 1])


class XGBRegR2Score(XGBRegressor):
    """
    Wrapper around an XGBoost regressor that has a non-standard score: For each
    output dimension it calculates the r-squared.
    """

    def __init__(self, n_estimators=500, max_depth=9, learning_rate=0.05, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, **kwargs)

    def score(self, X, y=None):
        y_pred = self.predict(X)
        n_outcome = y_pred.shape[1]
        result = np.zeros(shape=(n_outcome,))
        for out_idx in range(n_outcome):
            result[out_idx] = sklearn.metrics.r2_score(y[:, out_idx], y_pred[:, out_idx])
        return result


def balanced_downsample(X: pd.DataFrame, y: np.array, T_feat: pd.DataFrame, t_id: np.array):
    """
    Balanced down-sampling based on treatment identity

    :param X: Nuisance data to downsample
    :param y: Outcomes to downsample
    :param T_feat: features of the treatments
    :param t_id: which treatment was applied
    :return: Tuple of the down-sampled input variables
    """
    # Keep all the positive experiences
    pos_ex = y > 0
    positive_df = X[pos_ex].reset_index(drop=True)
    positive_actions = t_id[pos_ex]
    act_feature_positve = T_feat[pos_ex].reset_index(drop=True)
    y_positive = y[pos_ex]
    positive_df['action_id'] = positive_actions
    positive_df['y'] = y_positive
    positive_df = pd.concat([positive_df, act_feature_positve], axis=1).reset_index(drop=True)

    negative_df = X[~pos_ex].reset_index(drop=True)
    negative_actions = t_id[~pos_ex]
    act_feature_negative = T_feat[~pos_ex].reset_index(drop=True)
    y_negative = y[~pos_ex]

    negative_df['action_id'] = negative_actions
    negative_df['y'] = y_negative
    negative_df = pd.concat([negative_df, act_feature_negative], axis=1).reset_index(drop=True)

    df_downsampled = negative_df.groupby('action_id', group_keys=False).apply(lambda x: x.sample(frac=downsample_ratio))

    logger.info(f'Dowsampled negative examples by {downsample_ratio}')
    actions_before = negative_df['action_id'].value_counts()
    actions_after = df_downsampled['action_id'].value_counts()
    logger.info(f'Before: {actions_before}')
    logger.info(f'After: {actions_after}')

    # Final sample
    combo_df = pd.concat([positive_df, df_downsampled], axis=0).reset_index(drop=True)
    # this shuffles it
    combo_df = combo_df.sample(frac=1).reset_index(drop=True)
    X = combo_df[X.columns.values]
    T_feat = combo_df[T_feat.columns.values]
    y = combo_df['y'].to_numpy()
    t_id = combo_df['action_id'].to_numpy()

    n_pos_ex2 = y.sum()
    logger.info(f"After downsampling non-conversion, Conversion rate  now {n_pos_ex2/len(y)}")
    logger.info(f"X {X.shape}")
    logger.info(f"T {T_feat.shape}")
    logger.info(f"y {y.shape}")

    return X, y, T_feat, t_id


def marketing_dml_test(
    file_name: str,
    model_name: str,
    cv_folds: int = 2,
    collinearity_thresh: float = 0.5,
    downsample_ratio: float = 0.1,
    log_scale_skew_thresh: float | int = 3,
    dml_cv_params: str | None = None,
    y_cv_params: str | None = None,
    t_cv_params: str | None = None,
    val_pcnt: float = 0.2,
    do_evaluate: bool = False
):
    """
    Main function for testing DML on high dimensional marketing data sets. Adds pre-processing
    like scaling and removing highly collinear features. It can cross validates parameters of either
    first stage model, or the final model.  Saves out the scores of every model to a file.

    :param file_name: File that contains the data, assumed to be joblib
    :param model_name: Name of the model to use, either SparseLinearDML or CausalForestDML
    :param cv_folds: Number of CV folds to use in the DML algorithm
    :param collinearity_thresh: Collinearity threshold for removing collinear features
    :param downsample_ratio: Ratio to downsample non-positive examles
    :param log_scale_skew_thresh: Threshold on skew to employ log scaling
    :param dml_cv_params: Dict-as-string CV params for the final stage model
    :param y_cv_params: Dict-as-string CV params for the first stage outcome model
    :param t_cv_params: Dict-as-string CV params for the first stage treatment model

    """
    dml_cv_param_dict = json.loads(dml_cv_params) if dml_cv_params is not None else {}
    y_cv_param_dict = json.loads(y_cv_params) if y_cv_params is not None else {}
    t_cv_param_dict = json.loads(t_cv_params) if t_cv_params is not None else {}

    root_dir = os.environ.get("LOCAL_ANALYSIS_DIR", ".")
    logger.info(f"Using root dir {root_dir} loading data file {file_name}")
    data_file = os.path.join(os.path.abspath(root_dir), file_name)
    data = joblib.load(data_file)

    # This is specific to Offerfit - names of the data are derived from contextual bandits
    t_id = data['action_ids']
    X = data['X_no_act']
    T_feat = data['a_processed']
    y = data['real_reward']

    treat_df = pd.DataFrame(T_feat)
    treat_df['treat_id'] = t_id
    treat_df = treat_df.drop_duplicates().sort_values(by='treat_id').reindex()

    y[y > 1] = 1

    logger.info(f"t_id {t_id.shape}")
    logger.info(f"X {X.shape}")
    logger.info(f"T_feat {T_feat.shape}")
    logger.info(f"y {y.shape}")

    if 0 < downsample_ratio < 1:
        (X, y, T, _) = balanced_downsample(X=X, y=y, T_feat=T_feat, t_id=t_id)
    else:
        logger.info(f"Not downsampling data, had downsample_ratio = {downsample_ratio}")
        T = T_feat

    if log_scale_skew_thresh > 0:
        (X, T) = scale_marketing_data(X=X, T=T, log_scale_skew_thresh=log_scale_skew_thresh)
    else:
        logger.info(f"Not scaling, log_scale_skew_thresh = {log_scale_skew_thresh}")

    if 0 < collinearity_thresh < 1:
        X, T = remove_collinear_features(X=X, y=y, T=T, collinearity_thresh=collinearity_thresh)
    else:
        logger.info(f"Not checking collinearity, collinearity_thresh = {collinearity_thresh}")

    dtstr = datetime.now().strftime("%Y%m%dT%H%M%S")
    result_file = data_file.replace('.joblib', f'_{model_name}_results_{dtstr}.csv')
    if not os.path.isfile(result_file):
        with open(result_file, 'w') as output_results:
            for k in dml_cv_param_dict.keys():
                output_results.write(f"DML_{k},")
            for k in y_cv_param_dict.keys():
                output_results.write(f"Y_{k},")
            for k in t_cv_param_dict.keys():
                output_results.write(f"T_{k},")
            for f_idx in range(0, cv_folds):
                output_results.write(f"F{f_idx}_y_score,")
            for f_idx in range(0, cv_folds):
                for tidx in range(T.shape[1]):
                    output_results.write(f"F{f_idx}_T{tidx}_score,")
            output_results.write("Yresid_RMSE,Yresid_r2,Yresid_corc,Yresid_corp\n")

    output_results = open(result_file, 'a')

    # treat_est_df = treat_df.drop('treat_id', axis=1)
    # Effect of the treatments trained on
    x_cols = list(X.columns.values)
    t_cols = list(treat_df.columns.values)
    t_cols.remove('treat_id')
    X['x_id']=X.index.values
    treat_est_combo = X.merge(treat_df, how='cross')
    logger.info(f"Calculating effects of {len(treat_df)} treatments on audience size {len(X)} "
                f" total {len(treat_est_combo)} combinations")
    # Save the combined sample identifiers, and then remove them
    treat_identifiers = treat_est_combo[['x_id','treat_id']]
    treat_est_combo = treat_est_combo.drop(['x_id','treat_id'],axis=1)
    X = X.drop(['x_id'],axis=1)

    # Simple un-stratified train/test split - need to redo this ASAP

    if do_evaluate:
        ind = np.array(range(len(X)))
        train_ind = np.random.choice(len(X), int((1.0-val_pcnt)*len(X)), replace=False)
        val_ind = ind[~np.isin(ind, train_ind)]
        Xtrain = X.iloc[train_ind,:]
        Ttrain = T.iloc[train_ind,:]
        ytrain = y[train_ind]
        Xval = X.iloc[val_ind,:]
        Tval = T.iloc[val_ind,:]
        yval = y[val_ind]
    else:
        Xtrain = X
        Ttrain = T
        ytrain = y


    for y_param_combo in product(*y_cv_param_dict.values()):
        y_params = dict(zip(y_cv_param_dict.keys(), y_param_combo))

        logger.info("#############################################################################")

        # logger.info(f"Fitting the standard (confounded) model with params {y_params}")
        # noncausal_est = XGBClassAUCScore(**y_params)
        # noncausal_est.fit(pd.concat([X,T], axis=1), y)
        # logger.info("Standard model fit done")
        # standard_predictions = pd.DataFrame(noncausal_est.predict_proba(treat_est_combo)[:,1])
        # standard_predictions = pd.concat([standard_predictions,treat_identifiers],axis=1)
        # logger.info(f"Non-causal model fit is done")
        # logger.info(f'{pd.DataFrame(standard_predictions).describe()}')

        for t_param_combo in product(*t_cv_param_dict.values()):
            t_params = dict(zip(t_cv_param_dict.keys(), t_param_combo))

            for dml_param_combo in product(*dml_cv_param_dict.values()):
                # Map the combination back to the parameter names
                dml_params = dict(zip(dml_cv_param_dict.keys(), dml_param_combo))

                logger.info("---------------------------------------------------------------------")
                logger.info(
                    f"Starting {model_name}.fit dml_params = {dml_params}"
                    f", Y params = {y_params}, T params = {t_params}"
                )

                for parameter in dml_param_combo:
                    output_results.write(f"{parameter},")
                for parameter in y_param_combo:
                    output_results.write(f"{parameter},")
                for parameter in t_param_combo:
                    output_results.write(f"{parameter},")

                if model_name == "SparseLinearDML":
                    est = SparseLinearDML(
                        model_t=XGBRegR2Score(**t_params),
                        model_y=XGBClassAUCScore(**y_params),
                        discrete_outcome=True,
                        diagnostic_level=2,
                        cv=cv_folds,
                        **dml_params,
                    )
                elif model_name == "CausalForestDML":
                    est = CausalForestDML(
                        model_t=XGBRegR2Score(**t_params),
                        model_y=XGBClassAUCScore(**y_params),
                        discrete_outcome=True,
                        cv=cv_folds,
                        **dml_params,
                    )
                else:
                    raise ValueError(f"Ba")

                est.fit(ytrain, T=Ttrain, X=Xtrain, W=None)

                for fold_score in est.nuisance_scores_y[0]:
                    output_results.write(f"{fold_score},")
                for fold_scores in est.nuisance_scores_t[0]:
                    for score in fold_scores:
                        output_results.write(f"{score},")

                for y_resid_score in est.score_:
                    output_results.write(f"{y_resid_score},")
                output_results.write("\n")

                if do_evaluate:
                    logger.info("Fitting DRTester")
                    my_dr_tester = DRTester(
                        model_regression=XGBRegressor(**t_params),
                        model_propensity=XGBClassifier(**y_params),
                        cate=est,
                        cv=KFold(5)
                    )
                    my_dr_tester.fit_nuisance(Xval, Tval, yval, Xtrain, Ttrain, ytrain)
                    logger.info("Evaluating DRTester")
                    res = my_dr_tester.evaluate_all(Xval,Xtrain)
                    logger.info(f"{res.summary()}")

                logger.info(f"{model_name}.fit done, score: {est.score_}")
                # logger.info(f"Calculating effects on {treat_est_combo}")
                # effects = pd.DataFrame(est.effect(treat_est_combo[x_cols],
                #                     T1=treat_est_combo[t_cols]))
                # effects = pd.concat([effects, treat_identifiers],axis=1)
                # effects = effects.merge(standard_predictions,on=['x_id','treat_id'])
                # logger.info("effects done")
                # logger.info(f'{pd.DataFrame(effects).describe()}')


                if model_name == "SparseLinearDML":
                    logger.info(f"Coef:")
                    logger.info(est.coef_)
                    coef_file = result_file.replace('_results.csv', '_coef.csv')
                    np.savetxt(est.coef_, coef_file, fmt='%.6e', delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="", default="offerfit.joblib")
    parser.add_argument("--model_name", type=str, help="", default="CausalForestDML")
    parser.add_argument("--dml_cv_params", type=str, help="", default=None)
    parser.add_argument(
        "--y_cv_params", type=str, help="", default='{"max_depth":[9],"n_estimators":[500],"learning_rate":[0.01]}'
    )
    parser.add_argument(
        "--t_cv_params", type=str, help="", default='{"max_depth":[9],"n_estimators":[500],"learning_rate":[0.01]}'
    )

    parser.add_argument("--coll_thresh", type=float, help="", default=0.5)
    parser.add_argument("--downsample", type=float, help="", default=0.5)
    parser.add_argument("--skew_thresh", type=float, help="", default=3)
    parser.add_argument("--cv_folds", type=int, help="", default=2)

    args = parser.parse_args(sys.argv[1:])
    data_file = args.data_file
    model_name = args.model_name
    downsample_ratio = args.downsample
    collinearity_thresh = args.coll_thresh
    skew_thresh = args.skew_thresh
    dml_cv_params = args.dml_cv_params
    y_cv_params = args.y_cv_params
    t_cv_params = args.t_cv_params
    cv_folds = args.cv_folds

    # offerfit_dml_test("offerfit_marketing_test_data_20241002.joblib", "SparseLinearDML")
    marketing_dml_test(
        file_name=data_file,
        model_name=model_name,
        collinearity_thresh=collinearity_thresh,
        downsample_ratio=downsample_ratio,
        log_scale_skew_thresh=skew_thresh,
        dml_cv_params=dml_cv_params,
        y_cv_params=y_cv_params,
        t_cv_params=t_cv_params,
        cv_folds=cv_folds,
    )
