"""
evaluator.py — Comprehensive model evaluation for ChurnScope.

Covers: ROC-AUC, PR-AUC, F1, threshold optimisation (F1/cost/recall),
confusion matrix, SHAP feature importance, business metric reporting.
"""
from __future__ import annotations
import json, os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
from src import get_logger
logger = get_logger(__name__)

class ChurnEvaluator:
    def __init__(self, config: dict):
        self.config           = config
        self.threshold_cfg    = config.get("threshold",{})
        self.optimal_threshold = self.threshold_cfg.get("default",0.5)
        self.evaluation_results: Dict[str,Any] = {}

    def evaluate(self, y_true, y_scores, threshold=None,
                 feature_names=None, model=None, X_test=None) -> Dict[str,Any]:
        logger.info("Running model evaluation...")
        if threshold is None and self.threshold_cfg.get("optimize",True):
            threshold = self.optimize_threshold(y_true,y_scores)
        threshold = threshold or 0.5
        self.optimal_threshold = threshold
        y_pred = (y_scores >= threshold).astype(int)

        results = {
            "threshold": threshold,
            "roc_auc":   float(roc_auc_score(y_true,y_scores)),
            "pr_auc":    float(average_precision_score(y_true,y_scores)),
            "f1":        float(f1_score(y_true,y_pred)),
            "precision": float(precision_score(y_true,y_pred)),
            "recall":    float(recall_score(y_true,y_pred)),
        }
        tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
        results["confusion_matrix"] = {"true_negative":int(tn),"false_positive":int(fp),
                                        "false_negative":int(fn),"true_positive":int(tp)}
        results["business_metrics"]     = self._compute_business_metrics(tn,fp,fn,tp,y_true,y_scores)
        results["classification_report"] = classification_report(y_true,y_pred,
            target_names=["Non-churner","Churner"],output_dict=True)

        fpr,tpr,_ = roc_curve(y_true,y_scores)
        idx = np.linspace(0,len(fpr)-1,min(100,len(fpr))).astype(int)
        results["roc_curve"] = {"fpr":fpr[idx].tolist(),"tpr":tpr[idx].tolist()}

        prec_v,rec_v,_ = precision_recall_curve(y_true,y_scores)
        pr_idx = np.linspace(0,len(prec_v)-1,min(100,len(prec_v))).astype(int)
        results["pr_curve"] = {"precision":prec_v[pr_idx].tolist(),"recall":rec_v[pr_idx].tolist()}

        if model is not None and X_test is not None and feature_names is not None:
            shap_r = self._compute_shap(model,X_test,feature_names)
            if shap_r: results["feature_importance"] = shap_r

        self.evaluation_results = results
        self._log_summary(results)
        return results

    def optimize_threshold(self, y_true, y_scores) -> float:
        strategy = self.threshold_cfg.get("strategy","f1")
        prec_v,rec_v,thresholds = precision_recall_curve(y_true,y_scores)
        logger.info(f"Optimising threshold: strategy={strategy!r}")
        if strategy == "f1":
            f1s = np.where((prec_v+rec_v)>0, 2*prec_v*rec_v/(prec_v+rec_v), 0)
            best = int(np.argmax(f1s[:-1]))
            opt  = float(thresholds[best])
            logger.info(f"F1-optimal: {opt:.4f} (F1={f1s[best]:.4f})")
        elif strategy == "custom_cost":
            cc   = self.threshold_cfg.get("cost_matrix",{})
            fn_c,fp_c = cc.get("fn_cost",5),cc.get("fp_cost",1)
            costs = []
            for t in thresholds:
                yp = (y_scores>=t).astype(int)
                _,fp_n,fn_n,_ = confusion_matrix(y_true,yp).ravel()
                costs.append(fn_c*fn_n + fp_c*fp_n)
            best = int(np.argmin(costs)); opt = float(thresholds[best])
            logger.info(f"Cost-optimal: {opt:.4f}")
        elif strategy == "recall":
            eligible = rec_v[:-1] >= 0.80
            if eligible.any():
                best = int(np.argmax(prec_v[:-1][eligible]))
                opt  = float(thresholds[eligible][best])
            else:
                opt = float(thresholds[int(np.argmax(rec_v[:-1]))])
            logger.info(f"Recall-optimal: {opt:.4f}")
        else:
            opt = 0.5
        return opt

    def save_results(self, path: str = "models/artifacts/evaluation_results.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"w") as f: json.dump(self.evaluation_results,f,indent=2,default=str)
        logger.info(f"Evaluation results saved: {path}")

    def _compute_business_metrics(self,tn,fp,fn,tp,y_true,y_scores) -> Dict[str,Any]:
        avg_rev,offer_cost,ret_rate = 65.0,15.0,0.40
        total_churners   = int(y_true.sum())
        revenue_saved    = tp*avg_rev*ret_rate
        offer_cost_total = (tp+fp)*offer_cost
        return {
            "total_churners_in_test":int(total_churners),
            "churners_identified":int(tp),"churners_missed":int(fn),
            "non_churners_targeted":int(fp),
            "churn_coverage_pct":round(tp/total_churners*100,1) if total_churners else 0,
            "targeting_efficiency_pct":round(tp/(tp+fp)*100,1) if (tp+fp) else 0,
            "estimated_revenue_saved_usd":round(revenue_saved,2),
            "retention_offer_cost_usd":round(offer_cost_total,2),
            "net_business_benefit_usd":round(revenue_saved-offer_cost_total,2),
            "revenue_lost_without_model_usd":round(total_churners*avg_rev,2),
        }

    def _compute_shap(self,model,X_test,feature_names,max_samples=500):
        try:
            import shap
            n   = min(max_samples,X_test.shape[0])
            idx = np.random.choice(X_test.shape[0],n,replace=False)
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X_test[idx])
            sv  = sv[1] if isinstance(sv,list) else sv
            mean_abs = np.abs(sv).mean(axis=0)
            df = pd.DataFrame({"feature":feature_names[:len(mean_abs)],"mean_abs_shap":mean_abs})
            df = df.sort_values("mean_abs_shap",ascending=False)
            logger.info("Top 5 SHAP features: " + ", ".join(df["feature"].head(5).tolist()))
            return df.to_dict(orient="records")
        except ImportError:
            logger.warning("SHAP not installed — skipping.")
            return None
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return None

    def _log_summary(self, r):
        cm,biz = r["confusion_matrix"],r["business_metrics"]
        logger.info("="*55)
        logger.info("EVALUATION SUMMARY")
        logger.info(f"  Threshold  : {r['threshold']:.4f}")
        logger.info(f"  ROC-AUC    : {r['roc_auc']:.4f}")
        logger.info(f"  PR-AUC     : {r['pr_auc']:.4f}")
        logger.info(f"  F1         : {r['f1']:.4f}")
        logger.info(f"  Precision  : {r['precision']:.4f}")
        logger.info(f"  Recall     : {r['recall']:.4f}")
        logger.info(f"  TP/FP/FN/TN: {cm['true_positive']}/{cm['false_positive']}/{cm['false_negative']}/{cm['true_negative']}")
        logger.info(f"  Coverage   : {biz['churn_coverage_pct']}%")
        logger.info(f"  Net benefit: ${biz['net_business_benefit_usd']:,.2f}")
        logger.info("="*55)
