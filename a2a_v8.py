"""
A2A V8: Gemini + LangGraph Expert System for EDA, Monotonicity, Confounders, Simulation and Repair

Run:
    pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit networkx openpyxl google-genai langgraph typing_extensions
    streamlit run A2A_V8_Gemini_LangGraph_Expert_System.py

Optional Gemini key:
    Windows:
        setx GEMINI_API_KEY "YOUR_KEY"
    Mac/Linux:
        export GEMINI_API_KEY="YOUR_KEY"

Core idea:
    - Raw-data-first, not LLM-first.
    - Rules: increasing, decreasing, bidirectional.
    - Each KPI is checked in target-ordered bins.
    - Monotonic breaks are classified as:
        1. True break needing segmentation/repair
        2. Confounded break where another correctly-moving KPI explains the behavior
    - ML scenarios can vary KPIs, predict target, check global + local validity, and repair invalid KPI values to nearest valid raw value.
    - Gemini is optional and only polishes deterministic answers.

Sample questions:
    can you give me RoofArea
    Is RoofArea having good correlation with HeatingLoad?
    Show target ordered monotonic break analysis
    Show target ordered monotonic break analysis for RoofArea
    Which break cases are explained by confounders?
    Show violation range for RoofArea
    Show safe range for RelativeCompactness
    Show raw training data where RelativeCompactness is between 0.7 and 0.79
    If RelativeCompactness is between 0.7 and 0.79, what is expected HeatingLoad?
    If HeatingLoad target is between 15 and 20, tell me KPI ranges that respect business rules
    If HeatingLoad target is between 15 and 20, tell me KPI ranges that violates business rules
    Why is HeatingLoad high when RelativeCompactness is between 0.7 and 0.79?
    Repair RelativeCompactness value 0.76 to nearest valid value
    Simulate RelativeCompactness=0.76 RoofArea=150 GlazingArea=0.25 and repair invalid KPIs
"""

import os
import re
import json
import zipfile
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, TypedDict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import partial_dependence
from xgboost import XGBRegressor
from langgraph.graph import StateGraph, END

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
LOCAL_ZIP = "energy_efficiency.zip"
LOCAL_XLSX = "ENB2012_data.xlsx"

FEATURE_RENAME = {
    "X1": "RelativeCompactness",
    "X2": "SurfaceArea",
    "X3": "WallArea",
    "X4": "RoofArea",
    "X5": "OverallHeight",
    "X6": "Orientation",
    "X7": "GlazingArea",
    "X8": "GlazingAreaDistribution",
    "Y1": "HeatingLoad",
    "Y2": "CoolingLoad",
}

TARGET_OPTIONS = ["HeatingLoad", "CoolingLoad"]

# Change rules here for your own business case.
BUSINESS_RULES = {
    "RelativeCompactness": "decreasing",
    "SurfaceArea": "increasing",
    "WallArea": "increasing",
    "RoofArea": "increasing",
    "OverallHeight": "bidirectional",
    "Orientation": "bidirectional",
    "GlazingArea": "increasing",
    "GlazingAreaDistribution": "bidirectional",
}

SAMPLE_QUESTIONS = [
    "can you give me RoofArea",
    "What are the business rules?",
    "Is RoofArea having good correlation with HeatingLoad?",
    "Show EDA pattern summary",
    "Show target ordered monotonic break analysis",
    "Show target ordered monotonic break analysis for RoofArea",
    "Which break cases are explained by confounders?",
    "Show monotonic violation summary",
    "Show violation range for RoofArea",
    "Show safe range for RelativeCompactness",
    "Show raw training data where RelativeCompactness is between 0.7 and 0.79",
    "If RelativeCompactness is between 0.7 and 0.79, what is expected HeatingLoad?",
    "If HeatingLoad target is between 15 and 20, tell me KPI ranges that respect business rules",
    "If HeatingLoad target is between 15 and 20, tell me KPI ranges that violates business rules",
    "Why is HeatingLoad high when RelativeCompactness is between 0.7 and 0.79?",
    "Repair RelativeCompactness value 0.76 to nearest valid value",
    "Simulate RelativeCompactness=0.76 RoofArea=150 GlazingArea=0.25 and repair invalid KPIs",
]


# =========================================================
# STATE
# =========================================================

@dataclass
class AgentState:
    df: Optional[pd.DataFrame] = None
    target_col: str = "HeatingLoad"
    rules: Dict[str, str] = field(default_factory=dict)
    eda_report: Dict[str, Any] = field(default_factory=dict)
    monotonic_report: Dict[str, Any] = field(default_factory=dict)
    model_report: Dict[str, Any] = field(default_factory=dict)
    shap_report: Dict[str, Any] = field(default_factory=dict)
    graph: Optional[nx.MultiDiGraph] = None
    model: Any = None
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None


class GraphQAState(TypedDict, total=False):
    question: str
    app_state: AgentState
    intent: str
    final_answer: str


# =========================================================
# GEMINI
# =========================================================

class GeminiClient:
    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None

    def load(self) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError("Install google-genai: pip install google-genai")
        if not self.api_key:
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or enter it in sidebar.")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        try:
            if self.client is None:
                self.load()
            safe_prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            response = self.client.models.generate_content(model=self.model_name, contents=safe_prompt)
            return (response.text or "").strip()
        except Exception as exc:
            return f"Gemini explanation unavailable: {exc}"


def gemini_polish_answer(gemini: GeminiClient, question: str, deterministic_answer: str) -> str:
    prompt = f"""
Rewrite the deterministic answer in a clear business-friendly way.
Do not invent numbers. Do not change counts, ranges, or conclusions.
Preserve tables/code blocks if present.

Question:
{question}

Deterministic answer:
{deterministic_answer}
"""
    polished = gemini.generate(prompt)
    if polished.startswith("Gemini explanation unavailable"):
        return deterministic_answer
    return polished


# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data(show_spinner=False)
def load_energy_efficiency_data() -> pd.DataFrame:
    if not os.path.exists(LOCAL_XLSX):
        if not os.path.exists(LOCAL_ZIP):
            urllib.request.urlretrieve(UCI_ZIP_URL, LOCAL_ZIP)
        with zipfile.ZipFile(LOCAL_ZIP, "r") as zf:
            zf.extractall(".")

    df = pd.read_excel(LOCAL_XLSX)
    df = df.rename(columns=FEATURE_RENAME)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    expected_cols = list(BUSINESS_RULES.keys()) + TARGET_OPTIONS
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    return df[expected_cols].copy()


# =========================================================
# BASIC UTILITIES
# =========================================================

def normalize_text(s: str) -> str:
    return s.lower().replace("_", "").replace(" ", "")


def extract_feature(question: str, features: List[str]) -> Optional[str]:
    q_norm = normalize_text(question)
    for f in features:
        if normalize_text(f) in q_norm:
            return f
    # Also support common spelling variants.
    aliases = {
        "roof": "RoofArea",
        "roofarea": "RoofArea",
        "compactness": "RelativeCompactness",
        "relativecompactness": "RelativeCompactness",
        "glazing": "GlazingArea",
        "wall": "WallArea",
        "surface": "SurfaceArea",
        "height": "OverallHeight",
        "orientation": "Orientation",
    }
    for key, value in aliases.items():
        if key in q_norm and value in features:
            return value
    return None


def extract_range(question: str) -> Optional[Tuple[float, float]]:
    q = question.lower().replace(",", "")
    patterns = [
        r"between\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(-?\d+(?:\.\d+)?)",
        r"from\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return min(a, b), max(a, b)
    return None


def range_filter(df: pd.DataFrame, feature: str, low: float, high: float, tol: float = 1e-9) -> pd.DataFrame:
    temp = df.copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    return temp[(temp[feature] >= low - tol) & (temp[feature] <= high + tol)].copy()


def inside_ranges(value: float, ranges: List[Tuple[float, float]]) -> bool:
    return any(low <= value <= high for low, high in ranges)


def rule_direction_ok(rule: str, delta: float) -> bool:
    if rule == "increasing":
        return delta >= 0
    if rule == "decreasing":
        return delta <= 0
    return True


def movement_label(delta: float) -> str:
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "unchanged"


# =========================================================
# PIPELINE AGENTS
# =========================================================

class BaseAgent:
    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError


class DataLoaderAgent(BaseAgent):
    def __init__(self, target_col: str):
        self.target_col = target_col

    def run(self, state: AgentState) -> AgentState:
        state.df = load_energy_efficiency_data()
        state.target_col = self.target_col
        state.rules = BUSINESS_RULES.copy()
        return state


class DataQualityAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        df = state.df
        assert df is not None
        state.eda_report["data_quality"] = {
            "shape": df.shape,
            "missing": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }
        return state


class FeatureEDAAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        df = state.df
        assert df is not None
        features = list(state.rules.keys())
        target = state.target_col
        state.eda_report["correlations"] = (
            df[features + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False).to_dict()
        )
        state.eda_report["feature_summary"] = (
            df[features + [target]].describe().T.reset_index().rename(columns={"index": "feature"}).to_dict("records")
        )
        return state


class EDAPatternAgent(BaseAgent):
    @staticmethod
    def safe_bins(df: pd.DataFrame, feature: str, bins: int = 8) -> pd.Series:
        if feature in ["Orientation", "GlazingAreaDistribution", "OverallHeight"]:
            return df[feature].astype(str)
        try:
            return pd.qcut(df[feature], q=bins, duplicates="drop")
        except Exception:
            return pd.cut(df[feature], bins=bins)

    @staticmethod
    def alignment_from_diffs(diffs: np.ndarray, rule: str) -> Tuple[str, List[int]]:
        if rule == "increasing":
            violations = np.where(diffs < 0)[0].tolist()
        elif rule == "decreasing":
            violations = np.where(diffs > 0)[0].tolist()
        else:
            return "BIDIRECTIONAL_SEGMENT_BASED", []
        if not violations:
            return "GOOD_GLOBAL_ALIGNMENT", violations
        if len(violations) <= max(1, int(0.25 * max(len(diffs), 1))):
            return "MOSTLY_ALIGNED_WITH_LOCAL_VIOLATIONS", violations
        return "WEAK_OR_CONFLICTING_ALIGNMENT", violations

    def run(self, state: AgentState) -> AgentState:
        df = state.df
        assert df is not None
        target = state.target_col
        features = list(state.rules.keys())
        univariate = {}
        for feature in features:
            rule = state.rules[feature]
            temp = df[[feature, target]].copy()
            temp["bin"] = self.safe_bins(temp, feature)
            grouped = (
                temp.groupby("bin", observed=False)
                .agg(
                    feature_min=(feature, "min"),
                    feature_max=(feature, "max"),
                    feature_mean=(feature, "mean"),
                    target_mean=(target, "mean"),
                    target_median=(target, "median"),
                    count=(target, "size"),
                )
                .reset_index()
                .sort_values("feature_mean")
                .reset_index(drop=True)
            )
            diffs = np.diff(grouped["target_mean"].values)
            alignment, violations = self.alignment_from_diffs(diffs, rule)
            corr = float(df[[feature, target]].corr(numeric_only=True).iloc[0, 1])
            if rule == "bidirectional":
                rec = "Use segment-based interpretation; do not force monotonic correction."
            elif alignment == "GOOD_GLOBAL_ALIGNMENT":
                rec = "Business rule is globally supported."
            elif alignment == "MOSTLY_ALIGNED_WITH_LOCAL_VIOLATIONS":
                rec = "Mostly aligned; inspect local violation zones."
            else:
                rec = "Rule conflicts locally/globally; check confounders and segmentation."
            univariate[feature] = {
                "rule": rule,
                "correlation_with_target": corr,
                "business_rule_alignment": alignment,
                "violation_indices": violations,
                "recommendation": rec,
                "bin_summary": grouped.astype({"bin": "str"}).to_dict("records"),
            }

        corr_series = df[features + [target]].corr(numeric_only=True)[target].drop(target)
        top_features = corr_series.abs().sort_values(ascending=False).head(min(5, len(features))).index.tolist()
        interactions = []
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i + 1:]:
                try:
                    temp = df[[f1, f2, target]].copy()
                    temp["f1_bin"] = self.safe_bins(temp, f1, bins=4)
                    temp["f2_bin"] = self.safe_bins(temp, f2, bins=4)
                    grouped = temp.groupby(["f1_bin", "f2_bin"], observed=False)[target].agg(["mean", "count"]).reset_index()
                    spread = float(grouped["mean"].max() - grouped["mean"].min())
                    strength = spread / (float(df[target].std()) or 1.0)
                    pattern = "STRONG_INTERACTION" if strength >= 1.0 else "MODERATE_INTERACTION" if strength >= 0.5 else "WEAK_INTERACTION"
                    best = grouped.loc[grouped["mean"].idxmax()]
                    worst = grouped.loc[grouped["mean"].idxmin()]
                    interactions.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "pattern": pattern,
                        "target_mean_spread": spread,
                        "interaction_strength_vs_target_std": strength,
                        "lowest_target_segment": f"{f1}={worst['f1_bin']}, {f2}={worst['f2_bin']}",
                        "highest_target_segment": f"{f1}={best['f1_bin']}, {f2}={best['f2_bin']}",
                    })
                except Exception as exc:
                    interactions.append({"feature_1": f1, "feature_2": f2, "error": str(exc)})
        state.eda_report["pattern_agent"] = {
            "univariate_patterns": univariate,
            "multivariate_patterns": interactions,
            "top_multivariate_patterns": sorted(
                [x for x in interactions if "target_mean_spread" in x],
                key=lambda x: x["target_mean_spread"],
                reverse=True,
            )[:5],
            "features_needing_segmentation": [
                f for f, d in univariate.items()
                if d["business_rule_alignment"] in ["WEAK_OR_CONFLICTING_ALIGNMENT", "BIDIRECTIONAL_SEGMENT_BASED"]
            ],
        }
        return state


class EDAMonotonicityAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        df = state.df
        assert df is not None
        target = state.target_col
        report = {}
        for feature, rule in state.rules.items():
            temp = df[[feature, target]].copy()
            temp["bin"] = EDAPatternAgent.safe_bins(temp, feature)
            grouped = (
                temp.groupby("bin", observed=False)
                .agg(
                    feature_min=(feature, "min"),
                    feature_max=(feature, "max"),
                    feature_mean=(feature, "mean"),
                    target_mean=(target, "mean"),
                    count=(target, "size"),
                )
                .reset_index()
                .sort_values("feature_mean")
                .reset_index(drop=True)
            )
            diffs = np.diff(grouped["target_mean"].values)
            if rule == "increasing":
                violation_idx = np.where(diffs < 0)[0].tolist()
            elif rule == "decreasing":
                violation_idx = np.where(diffs > 0)[0].tolist()
            else:
                violation_idx = []
            details = []
            for idx in violation_idx:
                before, after = grouped.iloc[idx], grouped.iloc[idx + 1]
                details.append({
                    "from_bin": str(before["bin"]),
                    "to_bin": str(after["bin"]),
                    "from_target_mean": float(before["target_mean"]),
                    "to_target_mean": float(after["target_mean"]),
                    "change": float(after["target_mean"] - before["target_mean"]),
                })
            report[feature] = {
                "rule": rule,
                "bin_table": grouped.astype({"bin": "str"}).to_dict("records"),
                "check": {
                    "diffs": [float(x) for x in diffs],
                    "violation_indices": violation_idx,
                    "violation_count": len(violation_idx),
                    "violation_rate": round(len(violation_idx) / max(len(diffs), 1), 3),
                },
                "violation_details": details,
            }
        state.monotonic_report["eda_monotonicity"] = report
        return state


class ZoneRangeExtractor:
    @staticmethod
    def zone_bounds(z: Dict[str, Any]) -> Tuple[float, float]:
        return float(z["zone_feature_low"]), float(z["zone_feature_high"])

    @classmethod
    def extract(cls, rule_zones: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for feature, zr in rule_zones.items():
            safe, vio, info = [], [], []
            for z in zr.get("zones", []):
                low, high = cls.zone_bounds(z)
                if z["status"] == "RESPECTED":
                    safe.append((low, high))
                elif z["status"] == "VIOLATED":
                    vio.append((low, high))
                else:
                    info.append((low, high))
            out[feature] = {
                "rule": zr.get("rule"),
                "safe_ranges": safe,
                "violation_ranges": vio,
                "info_ranges": info,
                "safe_count": len(safe),
                "violation_count": len(vio),
                "info_count": len(info),
            }
        return out

    @staticmethod
    def to_dataframe(summary: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for f, data in summary.items():
            for label, ranges in [
                ("SAFE_RESPECTED", data.get("safe_ranges", [])),
                ("UNSAFE_VIOLATION", data.get("violation_ranges", [])),
                ("INFO_BIDIRECTIONAL", data.get("info_ranges", [])),
            ]:
                for low, high in ranges:
                    rows.append({"feature": f, "rule": data["rule"], "range_type": label, "low": low, "high": high})
        return pd.DataFrame(rows)


class RuleZoneAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        zone_report = {}
        for feature, result in state.monotonic_report.get("eda_monotonicity", {}).items():
            rule = result["rule"]
            bins = result["bin_table"]
            zones = []
            for i in range(len(bins) - 1):
                curr, nxt = bins[i], bins[i + 1]
                change = float(nxt["target_mean"] - curr["target_mean"])
                low = min(float(curr["feature_min"]), float(nxt["feature_min"]))
                high = max(float(curr["feature_max"]), float(nxt["feature_max"]))
                if rule == "increasing":
                    status = "RESPECTED" if change >= 0 else "VIOLATED"
                elif rule == "decreasing":
                    status = "RESPECTED" if change <= 0 else "VIOLATED"
                else:
                    status = "INFO"
                zones.append({
                    "feature": feature,
                    "rule": rule,
                    "zone_id": f"{feature}_zone_{i}_to_{i+1}",
                    "from_bin": curr["bin"],
                    "to_bin": nxt["bin"],
                    "zone_feature_low": low,
                    "zone_feature_high": high,
                    "from_feature_mean": float(curr["feature_mean"]),
                    "to_feature_mean": float(nxt["feature_mean"]),
                    "from_target_mean": float(curr["target_mean"]),
                    "to_target_mean": float(nxt["target_mean"]),
                    "target_change": change,
                    "status": status,
                })
            zone_report[feature] = {
                "rule": rule,
                "zones": zones,
                "respected_zones": [z for z in zones if z["status"] == "RESPECTED"],
                "violated_zones": [z for z in zones if z["status"] == "VIOLATED"],
                "info_zones": [z for z in zones if z["status"] == "INFO"],
            }
        state.monotonic_report["rule_zones"] = zone_report
        state.monotonic_report["range_summary"] = ZoneRangeExtractor.extract(zone_report)
        return state


class ModelTrainingAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        df = state.df
        assert df is not None
        features = list(state.rules.keys())
        X, y = df[features], df[state.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        state.model = model
        state.X_train, state.X_test, state.y_train, state.y_test = X_train, X_test, y_train, y_test
        state.model_report = {
            "target": state.target_col,
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "features": features,
        }
        return state


class PDPMonotonicityAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        X = state.X_test
        assert X is not None
        report = {}
        for feature, rule in state.rules.items():
            if rule == "bidirectional":
                continue
            try:
                idx = list(X.columns).index(feature)
                pdp = partial_dependence(state.model, X, features=[idx], grid_resolution=20)
                avg = pdp["average"][0]
                diffs = np.diff(avg)
                violations = np.where(diffs < 0)[0].tolist() if rule == "increasing" else np.where(diffs > 0)[0].tolist()
                report[feature] = {"rule": rule, "violation_count": len(violations), "violation_rate": round(len(violations) / max(len(diffs), 1), 3)}
            except Exception as exc:
                report[feature] = {"error": str(exc)}
        state.monotonic_report["pdp_monotonicity"] = report
        return state


class SHAPValidationAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        if not SHAP_AVAILABLE:
            state.shap_report = {"available": False, "message": "SHAP unavailable"}
            return state
        X = state.X_test.sample(min(200, len(state.X_test)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(state.model)
        values = explainer.shap_values(X)
        shap_df = pd.DataFrame(values, columns=X.columns, index=X.index)
        result = {"available": True, "features": {}}
        for feature, rule in state.rules.items():
            corr = np.corrcoef(X[feature], shap_df[feature])[0, 1]
            corr = 0.0 if np.isnan(corr) else float(corr)
            if rule == "increasing":
                status = "PASS" if corr >= 0 else "FAIL"
            elif rule == "decreasing":
                status = "PASS" if corr <= 0 else "FAIL"
            else:
                status = "INFO"
            result["features"][feature] = {
                "rule": rule,
                "status": status,
                "feature_shap_correlation": corr,
                "mean_abs_shap": float(np.abs(shap_df[feature]).mean()),
            }
        state.shap_report = result
        return state


class KnowledgeGraphAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        G = nx.MultiDiGraph()
        G.add_node("Dataset:EnergyEfficiency", type="dataset")
        G.add_node("Model:XGBoost", type="model", **state.model_report)
        G.add_edge("Dataset:EnergyEfficiency", "Model:XGBoost", relation="TRAINED_MODEL")
        for f, rule in state.rules.items():
            G.add_node(f"Feature:{f}", type="feature")
            G.add_node(f"Rule:{f}:{rule}", type="rule", rule=rule)
            G.add_edge(f"Feature:{f}", f"Rule:{f}:{rule}", relation="HAS_RULE")
        state.graph = G
        return state


# =========================================================
# SPECIALIST QA AGENTS
# =========================================================

class FeatureExplainerAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        feature = extract_feature(question, list(state.rules.keys()))
        if not feature:
            return None
        df = state.df
        assert df is not None
        s = df[feature]
        corr = state.eda_report.get("correlations", {}).get(feature, None)
        pattern = state.eda_report.get("pattern_agent", {}).get("univariate_patterns", {}).get(feature, {})
        range_summary = state.monotonic_report.get("range_summary", {}).get(feature, {})
        shap_info = state.shap_report.get("features", {}).get(feature, {}) if state.shap_report.get("available") else {}
        lines = [
            f"## KPI: `{feature}`",
            "",
            f"- Business rule: **{state.rules[feature]}**",
            f"- Raw data min: **{s.min():.3f}**",
            f"- Median: **{s.median():.3f}**",
            f"- Max: **{s.max():.3f}**",
        ]
        if corr is not None:
            strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak"
            lines.append(f"- Correlation with `{state.target_col}`: **{corr:.3f}** ({strength})")
        if pattern:
            lines.append(f"- Pattern alignment: **{pattern.get('business_rule_alignment')}**")
            lines.append(f"- Recommendation: {pattern.get('recommendation')}")
        if range_summary:
            lines.append(f"- Safe zones: **{range_summary.get('safe_count')}**")
            lines.append(f"- Violation zones: **{range_summary.get('violation_count')}**")
            lines.append(f"- Info zones: **{range_summary.get('info_count')}**")
        if shap_info:
            lines.append(f"- SHAP importance mean_abs_shap: **{shap_info.get('mean_abs_shap', 0):.3f}**")
            lines.append(f"- SHAP direction validation: **{shap_info.get('status')}**")
        lines.append("\n### Useful follow-up questions")
        lines.append(f"- `Show target ordered monotonic break analysis for {feature}`")
        lines.append(f"- `Show violation range for {feature}`")
        lines.append(f"- `Why is {state.target_col} high when {feature} is between <low> and <high>?`")
        return "\n".join(lines)


class RawDataQAAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not any(k in q for k in ["raw data", "training data", "actual rows", "show rows", "records", "rows where"]):
            return None
        feature = extract_feature(question, list(state.rules.keys()))
        rng = extract_range(question)
        if not feature or not rng:
            return None
        low, high = rng
        df = state.df
        subset = range_filter(df, feature, low, high)
        vals = sorted(pd.to_numeric(df[feature], errors="coerce").dropna().unique().tolist())
        inside = [v for v in vals if low <= v <= high]
        lines = [
            f"## Raw training data for `{feature}` between `{low}` and `{high}`",
            "",
            f"- Matching raw rows: **{len(subset)}**",
            f"- Available raw values inside range: `{inside}`",
        ]
        if subset.empty:
            nearest = sorted(vals, key=lambda x: min(abs(x - low), abs(x - high)))[:10]
            lines.append(f"- Nearest available raw values: `{nearest}`")
            return "\n".join(lines)
        t = subset[state.target_col]
        lines.extend([
            "",
            f"### `{state.target_col}` summary",
            f"- Mean: **{t.mean():.3f}**",
            f"- Median: **{t.median():.3f}**",
            f"- Min-Max: **{t.min():.3f} to {t.max():.3f}**",
            "",
            "### First 50 rows",
            "```text",
            subset.head(50).to_string(index=False),
            "```",
        ])
        return "\n".join(lines)


class EDAQAAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if any(k in q for k in ["business rules", "list rules", "what rules"]):
            return "## Business rules\n\n" + "\n".join([f"- **{f}** -> `{r}`" for f, r in state.rules.items()])
        if any(k in q for k in ["eda", "pattern", "segmentation", "need segmentation"]):
            pattern = state.eda_report.get("pattern_agent", {})
            uni = pattern.get("univariate_patterns", {})
            lines = ["## EDA Pattern Summary", "", "### Univariate rule alignment"]
            for f, d in uni.items():
                lines.append(f"- **{f}** | rule `{d['rule']}` | corr `{d['correlation_with_target']:.3f}` | alignment `{d['business_rule_alignment']}` | {d['recommendation']}")
            lines.append("\n### Features needing segmentation")
            needs = pattern.get("features_needing_segmentation", [])
            lines.extend([f"- **{f}**" for f in needs] if needs else ["- None detected"])
            lines.append("\n### Top interactions")
            for item in pattern.get("top_multivariate_patterns", []):
                lines.append(f"- **{item['feature_1']} + {item['feature_2']}** -> `{item['pattern']}` | spread `{item['target_mean_spread']:.3f}` | highest: {item['highest_target_segment']}")
            return "\n".join(lines)
        if "correlation" in q or "correlated" in q or "related" in q:
            feature = extract_feature(question, list(state.rules.keys()))
            corr = state.eda_report.get("correlations", {})
            if feature:
                value = corr.get(feature)
                if value is None:
                    return None
                strength = "strong" if abs(value) >= 0.7 else "moderate" if abs(value) >= 0.3 else "weak"
                direction = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                rule = state.rules[feature]
                aligned = ((rule == "increasing" and value > 0) or (rule == "decreasing" and value < 0) or rule == "bidirectional")
                return f"## Correlation check: `{feature}` vs `{state.target_col}`\n\n- Correlation: **{value:.3f}**\n- Strength: **{strength}**\n- Direction: **{direction}**\n- Business rule: `{rule}`\n- Rule alignment: **{'aligned' if aligned else 'not aligned'}**"
            return "## Correlations\n\n" + "\n".join([f"- **{f}**: `{v:.3f}`" for f, v in corr.items()])
        return None


class MonotonicQAAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        summary = state.monotonic_report.get("range_summary", {})
        mono = state.monotonic_report.get("eda_monotonicity", {})
        if any(k in q for k in ["monotonic summary", "violation summary", "which kpis violate", "rule summary"]):
            lines = ["## Monotonicity summary", ""]
            for f, r in mono.items():
                check = r.get("check", {})
                rs = summary.get(f, {})
                lines.append(f"- **{f}** | rule `{r['rule']}` | violations `{check.get('violation_count')}` | violation rate `{check.get('violation_rate')}` | safe zones `{rs.get('safe_count')}` | violation zones `{rs.get('violation_count')}`")
            return "\n".join(lines)
        wants_violation = any(k in q for k in ["violation range", "violated range", "unsafe", "show violation", "where violated"])
        wants_safe = any(k in q for k in ["safe range", "respected range", "show safe", "where respected"])
        if wants_violation or wants_safe:
            feature = extract_feature(question, list(summary.keys()))
            features = [feature] if feature else list(summary.keys())
            key = "violation_ranges" if wants_violation else "safe_ranges"
            label = "VIOLATION" if wants_violation else "SAFE"
            lines = [f"## {label} ranges", ""]
            for f in features:
                data = summary[f]
                lines.append(f"### {f} — rule `{data['rule']}`")
                ranges = data.get(key, [])
                if not ranges:
                    lines.append(f"- No {label.lower()} ranges found.")
                for low, high in ranges:
                    lines.append(f"- `{low:.3f}` to `{high:.3f}` -> **{label}**")
                lines.append("")
            return "\n".join(lines)
        if any(k in q for k in ["bidirectional", "bi directional", "non monotonic", "non-monotonic"]):
            lines = ["## Bidirectional behavior", ""]
            for f, rule in state.rules.items():
                if rule != "bidirectional":
                    continue
                bins = mono.get(f, {}).get("bin_table", [])
                lines.append(f"### {f}")
                for b in bins:
                    lines.append(f"- `{b['bin']}` -> mean `{state.target_col}` `{b['target_mean']:.3f}`, count `{b['count']}`")
            return "\n".join(lines)
        return None


class TargetOrderedBreakAgent:
    def __init__(self, n_bins: int = 8, confounder_strength_threshold: float = 0.25):
        self.n_bins = n_bins
        self.confounder_strength_threshold = confounder_strength_threshold

    def build_target_order_table(self, state: AgentState) -> pd.DataFrame:
        df = state.df.copy()
        target = state.target_col
        features = list(state.rules.keys())
        try:
            df["target_bin"] = pd.qcut(df[target], q=self.n_bins, duplicates="drop")
        except Exception:
            df["target_bin"] = pd.cut(df[target], bins=self.n_bins)
        grouped = (
            df.groupby("target_bin", observed=False)
            .agg(**{target: (target, "mean")}, **{f: (f, "mean") for f in features}, count=(target, "size"))
            .reset_index()
            .sort_values(target)
            .reset_index(drop=True)
        )
        grouped["target_bin"] = grouped["target_bin"].astype(str)
        return grouped

    def analyze(self, state: AgentState, requested_feature: Optional[str] = None) -> Dict[str, Any]:
        target = state.target_col
        features = list(state.rules.keys())
        table = self.build_target_order_table(state)
        analysis_features = [requested_feature] if requested_feature else features
        break_cases, summaries = [], []
        for feature in analysis_features:
            rule = state.rules.get(feature)
            if not rule:
                continue
            if rule == "bidirectional":
                summaries.append({"feature": feature, "rule": rule, "status": "BIDIRECTIONAL_SEGMENT_ONLY", "break_count": 0, "confounded_break_count": 0, "true_break_count": 0})
                continue
            local_breaks = []
            for i in range(len(table) - 1):
                curr, nxt = table.iloc[i], table.iloc[i + 1]
                target_delta = float(nxt[target] - curr[target])
                kpi_delta = float(nxt[feature] - curr[feature])
                if rule_direction_ok(rule, kpi_delta):
                    continue
                confounders = []
                for other in features:
                    if other == feature:
                        continue
                    other_rule = state.rules[other]
                    other_delta = float(nxt[other] - curr[other])
                    std = float(state.df[other].std()) or 1.0
                    strength = abs(other_delta) / std
                    if other_rule == "bidirectional":
                        is_confounder = strength >= self.confounder_strength_threshold
                        ctype = "BIDIRECTIONAL_SEGMENT_FACTOR"
                    else:
                        is_confounder = rule_direction_ok(other_rule, other_delta) and strength >= self.confounder_strength_threshold
                        ctype = "RULE_ALIGNED_CONFOUNDER" if is_confounder else "NON_ALIGNED_FACTOR"
                    if is_confounder:
                        confounders.append({"feature": other, "rule": other_rule, "movement": movement_label(other_delta), "delta": other_delta, "normalized_strength": strength, "confounder_type": ctype})
                confounders = sorted(confounders, key=lambda x: x["normalized_strength"], reverse=True)
                decision = "NO_CORRECTION_NEEDED_CONFOUNDER_EXPLAINS" if confounders else "TRUE_MONOTONIC_BREAK_NEEDS_SEGMENT_OR_REPAIR"
                item = {
                    "feature": feature,
                    "rule": rule,
                    "from_target_bin": curr["target_bin"],
                    "to_target_bin": nxt["target_bin"],
                    "from_target_mean": float(curr[target]),
                    "to_target_mean": float(nxt[target]),
                    "target_delta": target_delta,
                    "from_kpi_mean": float(curr[feature]),
                    "to_kpi_mean": float(nxt[feature]),
                    "kpi_delta": kpi_delta,
                    "kpi_movement": movement_label(kpi_delta),
                    "confounders": confounders[:5],
                    "decision": decision,
                }
                local_breaks.append(item)
                break_cases.append(item)
            true_breaks = [b for b in local_breaks if b["decision"] == "TRUE_MONOTONIC_BREAK_NEEDS_SEGMENT_OR_REPAIR"]
            confounded = [b for b in local_breaks if b["decision"] == "NO_CORRECTION_NEEDED_CONFOUNDER_EXPLAINS"]
            summaries.append({"feature": feature, "rule": rule, "status": "PASS" if not local_breaks else "HAS_BREAKS", "break_count": len(local_breaks), "confounded_break_count": len(confounded), "true_break_count": len(true_breaks)})
        return {"target_order_table": table.to_dict("records"), "feature_summaries": summaries, "break_cases": break_cases}

    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not any(k in q for k in ["target ordered", "target order", "arranged by target", "order of target", "monotonic break", "breaking case", "break case", "confounding", "confounder", "correction needed", "correction not needed"]):
            return None
        feature = extract_feature(question, list(state.rules.keys()))
        result = self.analyze(state, feature)
        lines = ["## Target-ordered monotonic break analysis", "", f"Target used for ordering: `{state.target_col}`", "Rows are grouped by target quantile bins, then KPI mean movement is checked as target increases.", "", "### KPI summary"]
        for s in result["feature_summaries"]:
            lines.append(f"- **{s['feature']}** | rule `{s['rule']}` | status `{s['status']}` | breaks `{s['break_count']}` | confounded `{s['confounded_break_count']}` | true breaks `{s['true_break_count']}`")
        if not result["break_cases"]:
            lines.append("\nNo monotonic break cases found.")
            return "\n".join(lines)
        lines.append("\n### Break cases and decisions")
        for b in result["break_cases"][:25]:
            lines.append(f"\n#### {b['feature']} break: `{b['from_target_bin']}` -> `{b['to_target_bin']}`")
            lines.append(f"- Target mean: `{b['from_target_mean']:.3f}` -> `{b['to_target_mean']:.3f}`")
            lines.append(f"- KPI mean: `{b['from_kpi_mean']:.3f}` -> `{b['to_kpi_mean']:.3f}` ({b['kpi_movement']}, delta `{b['kpi_delta']:.6f}`)")
            lines.append(f"- Decision: **{b['decision']}**")
            if b["confounders"]:
                lines.append("- Confounders explaining this break:")
                for c in b["confounders"]:
                    lines.append(f"  - **{c['feature']}** | rule `{c['rule']}` | moved `{c['movement']}` | strength `{c['normalized_strength']:.3f}` | `{c['confounder_type']}`")
            else:
                lines.append("- No strong rule-aligned confounder found. Candidate for segmentation/repair.")
        lines.append("\n### Interpretation")
        lines.append("- Confounded break: correction usually not needed because another KPI explains the target movement.")
        lines.append("- True break: no strong confounder found; consider segmentation or nearest-valid repair.")
        lines.append("- Bidirectional KPIs are context variables and are not globally repaired.")
        return "\n".join(lines)


class KPIToTargetAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        feature = extract_feature(question, list(state.rules.keys()))
        rng = extract_range(question)
        q = question.lower()
        if not feature or not rng:
            return None
        if not any(k in q for k in ["expected", "predict", "prediction", "target", "heatingload", "coolingload", "final value"]):
            return None
        low, high = rng
        subset = range_filter(state.df, feature, low, high)
        if subset.empty:
            values = sorted(pd.to_numeric(state.df[feature], errors="coerce").dropna().unique().tolist())
            nearest = sorted(values, key=lambda x: min(abs(x - low), abs(x - high)))[:10]
            return f"No raw rows found for `{feature}` between `{low}` and `{high}`. Nearest available values: `{nearest}`"
        features = list(state.rules.keys())
        subset = subset.copy()
        subset["prediction"] = state.model.predict(subset[features])
        pred, actual = subset["prediction"], subset[state.target_col]
        rs = state.monotonic_report.get("range_summary", {}).get(feature, {})
        safe_overlap = any(max(low, a) <= min(high, b) for a, b in rs.get("safe_ranges", []))
        vio_overlap = any(max(low, a) <= min(high, b) for a, b in rs.get("violation_ranges", []))
        return f"## Expected `{state.target_col}` for `{feature}` between `{low}` and `{high}`\n\n- Matching raw rows: **{len(subset)}**\n- Business rule: `{state.rules[feature]}`\n- Safe zone overlap: **{safe_overlap}**\n- Violation zone overlap: **{vio_overlap}**\n\n### Model prediction\n- Mean: **{pred.mean():.3f}**\n- Median: **{pred.median():.3f}**\n- P25-P75: **{pred.quantile(0.25):.3f} to {pred.quantile(0.75):.3f}**\n\n### Historical actual target\n- Mean: **{actual.mean():.3f}**\n- Median: **{actual.median():.3f}**\n- Min-Max: **{actual.min():.3f} to {actual.max():.3f}**"


class TargetToKPIAgent:
    @staticmethod
    def inside_safe(value: float, zones: Dict[str, Any]) -> bool:
        if zones.get("rule") == "bidirectional":
            return True
        return any(ZoneRangeExtractor.zone_bounds(z)[0] <= value <= ZoneRangeExtractor.zone_bounds(z)[1] for z in zones.get("respected_zones", []))

    @staticmethod
    def inside_violation(value: float, zones: Dict[str, Any]) -> bool:
        if zones.get("rule") == "bidirectional":
            return False
        return any(ZoneRangeExtractor.zone_bounds(z)[0] <= value <= ZoneRangeExtractor.zone_bounds(z)[1] for z in zones.get("violated_zones", []))

    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not ("target" in q or state.target_col.lower() in q or "heatingload" in q or "coolingload" in q):
            return None
        if not any(k in q for k in ["kpi", "feature", "features", "keep", "range"]):
            return None
        rng = extract_range(question)
        if not rng:
            return None
        tmin, tmax = rng
        violation_mode = any(k in q for k in ["violate", "violates", "violated", "violation", "unsafe", "risk"])
        features = list(state.rules.keys())
        sample = state.df[features].copy()
        sample["prediction"] = state.model.predict(sample)
        filtered = sample[(sample["prediction"] >= tmin) & (sample["prediction"] <= tmax)].copy()
        if filtered.empty:
            return f"No model-predicted rows found where `{state.target_col}` is between `{tmin}` and `{tmax}`."
        rule_zones = state.monotonic_report.get("rule_zones", {})
        if violation_mode:
            valid = filtered[filtered.apply(lambda row: any(self.inside_violation(float(row[f]), z) for f, z in rule_zones.items()), axis=1)].copy()
            mode = "VIOLATING"
        else:
            valid = filtered[filtered.apply(lambda row: all(self.inside_safe(float(row[f]), z) for f, z in rule_zones.items()), axis=1)].copy()
            mode = "RESPECTING"
        if valid.empty:
            return f"Rows matched target range, but none matched **{mode}** rule mode. Target-matched rows before rule filter: {len(filtered)}."
        top_features = get_top_shap_features(state, min(5, len(features)))
        lines = [f"## KPI ranges for `{state.target_col}` between `{tmin}` and `{tmax}`", "", f"- Mode: **{mode} business rules**", f"- Target-matched rows before rule filter: **{len(filtered)}**", f"- Rows after rule filter: **{len(valid)}**", f"- Showing top SHAP features: **{', '.join(top_features)}**", "", "### Recommended KPI ranges"]
        for f in top_features:
            s = valid[f]
            lines.append(f"- **{f}**: p25 `{s.quantile(0.25):.3f}` to p75 `{s.quantile(0.75):.3f}`, median `{s.median():.3f}`")
        return "\n".join(lines)


class ExpertReasoningAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not any(k in q for k in ["why", "cause", "causing", "driver", "drivers", "high", "low", "extreme", "other factor", "hidden factor"]):
            return None
        feature = extract_feature(question, list(state.rules.keys()))
        rng = extract_range(question)
        if not feature or not rng:
            return None
        low, high = rng
        subset = range_filter(state.df, feature, low, high)
        if subset.empty:
            vals = sorted(pd.to_numeric(state.df[feature], errors="coerce").dropna().unique().tolist())
            nearest = sorted(vals, key=lambda x: min(abs(x - low), abs(x - high)))[:10]
            return f"No raw rows found for `{feature}` between `{low}` and `{high}`. Nearest available values: `{nearest}`"
        target = state.target_col
        q25, q75 = subset[target].quantile(0.25), subset[target].quantile(0.75)
        low_group, high_group = subset[subset[target] <= q25], subset[subset[target] >= q75]
        if low_group.empty or high_group.empty:
            return f"Found {len(subset)} rows, but not enough high/low target rows to derive drivers."
        rows = []
        for f in state.rules.keys():
            if f == feature:
                continue
            low_mean, high_mean = float(low_group[f].mean()), float(high_group[f].mean())
            diff = high_mean - low_mean
            std = float(state.df[f].std()) or 1.0
            effect = abs(diff) / std
            rule = state.rules[f]
            if rule == "bidirectional":
                check = "BIDIRECTIONAL_SEGMENT_FACTOR"
            elif rule_direction_ok(rule, diff):
                check = "ALIGNED"
            else:
                check = "CONFLICTS_WITH_RULE"
            rows.append({"feature": f, "rule": rule, "low_mean": low_mean, "high_mean": high_mean, "diff": diff, "effect": effect, "rule_check": check})
        rows = sorted(rows, key=lambda x: x["effect"], reverse=True)
        lines = [f"## Expert reasoning for `{feature}` between `{low}` and `{high}`", "", f"- Raw rows found: **{len(subset)}**", f"- Mean `{target}`: **{subset[target].mean():.3f}**", f"- Low target threshold Q25: **{q25:.3f}**", f"- High target threshold Q75: **{q75:.3f}**", "", "### Top other factors separating high target from low target"]
        for r in rows[:5]:
            direction = "higher" if r["diff"] > 0 else "lower"
            lines.append(f"- **{r['feature']}** is **{direction}** in high-target rows | low mean `{r['low_mean']:.3f}` vs high mean `{r['high_mean']:.3f}` | rule `{r['rule']}` | check `{r['rule_check']}`")
        return "\n".join(lines)


class RepairAgent:
    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not any(k in q for k in ["repair", "impute", "make valid", "nearest valid", "fix violation", "correct violation"]):
            return None
        feature = extract_feature(question, list(state.rules.keys()))
        nums = re.findall(r"-?\d+(?:\.\d+)?", q)
        if not feature or not nums:
            return "Please ask like: `Repair RelativeCompactness value 0.76 to nearest valid value`."
        value = float(nums[-1])
        rule = state.rules[feature]
        raw_values = sorted(pd.to_numeric(state.df[feature], errors="coerce").dropna().unique().tolist())
        nearest_raw = min(raw_values, key=lambda x: abs(float(x) - value)) if raw_values else value
        summary = state.monotonic_report.get("range_summary", {}).get(feature, {})
        safe_ranges, violation_ranges = summary.get("safe_ranges", []), summary.get("violation_ranges", [])
        if rule == "bidirectional":
            return f"## Constraint repair for `{feature}`\n\n- Rule: `{rule}`\n- Original value: `{value}`\n- Nearest raw training value: **{nearest_raw}**\n- Repair verdict: **No unidirectional repair applied**\n\nUse segment-level expert reasoning."
        is_safe, is_violation = inside_ranges(value, safe_ranges), inside_ranges(value, violation_ranges)
        safe_values = [float(v) for v in raw_values if inside_ranges(float(v), safe_ranges)]
        if is_safe:
            nearest_valid, status = value, "ALREADY_VALID"
        elif safe_values:
            nearest_valid, status = min(safe_values, key=lambda x: abs(x - value)), "REPAIRED_TO_NEAREST_SAFE_RAW_VALUE"
        else:
            nearest_valid, status = float(nearest_raw), "FALLBACK_TO_NEAREST_RAW_VALUE_NO_SAFE_ZONE"
        return f"## Constraint repair for `{feature}`\n\n- Rule: `{rule}`\n- Original value: `{value}`\n- Status: **{status}**\n- Was in safe zone: **{is_safe}**\n- Was in violation zone: **{is_violation}**\n- Nearest raw training value: **{nearest_raw}**\n- Nearest valid value used: **{nearest_valid}**\n- Movement required: **{nearest_valid - value:.6f}**\n\n### Diagnostics\n- Safe ranges: `{safe_ranges}`\n- Violation ranges: `{violation_ranges}`"


class ScenarioSimulationRepairAgent:
    def parse_assignments(self, state: AgentState, question: str) -> Dict[str, float]:
        q = question.lower().replace(",", "")
        values = {}
        for feature in state.rules.keys():
            fname = feature.lower()
            patterns = [rf"{fname}\s*(?:=|is|as|value)?\s*(-?\d+(?:\.\d+)?)", rf"{fname}.*?(-?\d+(?:\.\d+)?)"]
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    values[feature] = float(m.group(1))
                    break
        return values

    def nearest_safe_value(self, state: AgentState, feature: str, value: float) -> Dict[str, Any]:
        raw_values = sorted(pd.to_numeric(state.df[feature], errors="coerce").dropna().unique().tolist())
        nearest_raw = min(raw_values, key=lambda x: abs(float(x) - value)) if raw_values else value
        summary = state.monotonic_report.get("range_summary", {}).get(feature, {})
        safe_ranges, violation_ranges = summary.get("safe_ranges", []), summary.get("violation_ranges", [])
        rule = state.rules[feature]
        if rule == "bidirectional":
            return {"feature": feature, "rule": rule, "original": value, "repaired": float(nearest_raw), "delta": float(nearest_raw) - value, "status": "BIDIRECTIONAL_NO_GLOBAL_REPAIR"}
        safe_values = [float(v) for v in raw_values if inside_ranges(float(v), safe_ranges)]
        is_safe = inside_ranges(value, safe_ranges)
        is_violation = inside_ranges(value, violation_ranges)
        if is_safe:
            repaired, status = float(value), "ALREADY_SAFE_GLOBAL"
        elif safe_values:
            repaired, status = min(safe_values, key=lambda x: abs(x - value)), "REPAIRED_TO_NEAREST_GLOBAL_SAFE_RAW_VALUE"
        else:
            repaired, status = float(nearest_raw), "FALLBACK_TO_NEAREST_RAW_VALUE_NO_SAFE_ZONE"
        return {"feature": feature, "rule": rule, "original": float(value), "repaired": repaired, "delta": repaired - float(value), "was_safe_global": bool(is_safe), "was_violation_global": bool(is_violation), "status": status}

    def local_bin_check(self, state: AgentState, scenario: pd.Series, predicted_target: float) -> Dict[str, Any]:
        ordered = TargetOrderedBreakAgent().build_target_order_table(state)
        target = state.target_col
        idx = int((ordered[target] - predicted_target).abs().idxmin())
        selected = ordered.iloc[idx]
        checks = []
        for feature, rule in state.rules.items():
            value = float(scenario[feature])
            mean = float(selected[feature])
            std = float(state.df[feature].std()) or 1.0
            dist = abs(value - mean) / std
            status = "LOCAL_OK" if dist <= 1.0 else "LOCAL_OUT_OF_BIN_PATTERN"
            if rule == "bidirectional":
                status = "BIDIRECTIONAL_CONTEXT_CHECK"
            checks.append({"feature": feature, "rule": rule, "scenario_value": value, "nearest_target_bin": selected["target_bin"], "bin_feature_mean": mean, "normalized_distance": dist, "local_status": status})
        return {"nearest_target_bin": selected["target_bin"], "nearest_bin_target_mean": float(selected[target]), "local_checks": checks}

    def simulate(self, state: AgentState, assignments: Dict[str, float]) -> Dict[str, Any]:
        features = list(state.rules.keys())
        base = state.df[features].median(numeric_only=True)
        scenario = base.copy()
        for f, v in assignments.items():
            scenario[f] = v
        before = float(state.model.predict(pd.DataFrame([scenario[features]]))[0])
        repaired = scenario.copy()
        repairs = []
        for f, v in assignments.items():
            rep = self.nearest_safe_value(state, f, float(v))
            repairs.append(rep)
            if rep["status"] not in ["ALREADY_SAFE_GLOBAL", "BIDIRECTIONAL_NO_GLOBAL_REPAIR"]:
                repaired[f] = rep["repaired"]
        after = float(state.model.predict(pd.DataFrame([repaired[features]]))[0])
        return {"input_assignments": assignments, "prediction_before_repair": before, "scenario_before_repair": scenario.to_dict(), "repairs": repairs, "prediction_after_repair": after, "scenario_after_repair": repaired.to_dict(), "local_check_before_repair": self.local_bin_check(state, scenario, before), "local_check_after_repair": self.local_bin_check(state, repaired, after)}

    def answer(self, state: AgentState, question: str) -> Optional[str]:
        q = question.lower()
        if not any(k in q for k in ["simulate", "vary", "what if", "scenario", "ml prediction", "prediction by varying"]):
            return None
        assignments = self.parse_assignments(state, question)
        if not assignments:
            return "Please provide KPI assignments. Example: `Simulate RelativeCompactness=0.76 RoofArea=150 GlazingArea=0.25 and repair invalid KPIs`."
        result = self.simulate(state, assignments)
        lines = ["## ML scenario simulation with global + local monotonic repair", "", f"- Input assignments: `{assignments}`", f"- Prediction before repair: **{result['prediction_before_repair']:.3f}**", f"- Prediction after repair: **{result['prediction_after_repair']:.3f}**", "", "### KPI repair decisions"]
        for r in result["repairs"]:
            lines.append(f"- **{r['feature']}** | rule `{r['rule']}` | original `{r['original']}` -> repaired `{r['repaired']}` | delta `{r['delta']:.6f}` | `{r['status']}`")
        lines.append("\n### Local target-bin check after repair")
        local = result["local_check_after_repair"]
        lines.append(f"- Nearest target bin: `{local.get('nearest_target_bin')}`")
        for c in local.get("local_checks", [])[:8]:
            lines.append(f"- **{c['feature']}** | value `{c['scenario_value']:.3f}` | bin mean `{c['bin_feature_mean']:.3f}` | distance `{c['normalized_distance']:.3f}` | `{c['local_status']}`")
        return "\n".join(lines)


# =========================================================
# LANGGRAPH ROUTER
# =========================================================

def classify_intent_node(gstate: GraphQAState) -> GraphQAState:
    q = gstate["question"].lower()
    state = gstate["app_state"]
    feature = extract_feature(q, list(state.rules.keys()))

    if any(k in q for k in ["simulate", "vary", "what if", "scenario", "ml prediction", "prediction by varying"]):
        intent = "scenario"
    elif any(k in q for k in ["target ordered", "target order", "arranged by target", "order of target", "monotonic break", "breaking case", "break case", "confounding", "confounder", "correction needed", "correction not needed"]):
        intent = "target_order"
    elif any(k in q for k in ["repair", "impute", "make valid", "nearest valid", "fix violation", "correct violation"]):
        intent = "repair"
    elif any(k in q for k in ["raw data", "training data", "actual rows", "show rows", "records", "rows where"]):
        intent = "raw"
    elif any(k in q for k in ["why", "cause", "causing", "driver", "drivers", "high", "low", "extreme", "other factor", "hidden factor"]):
        intent = "expert"
    elif ("target" in q or "heatingload" in q or "coolingload" in q) and any(k in q for k in ["kpi", "feature", "features", "keep", "range"]):
        intent = "target_to_kpi"
    elif any(k in q for k in ["expected", "predict", "prediction", "final value"]):
        intent = "kpi_to_target"
    elif any(k in q for k in ["monotonic", "violation", "safe", "bidirectional", "respected", "unsafe"]):
        intent = "monotonic"
    elif any(k in q for k in ["eda", "pattern", "univariate", "multivariate", "interaction", "correlation", "business rules", "segmentation"]):
        intent = "eda"
    elif feature:
        intent = "feature"
    else:
        intent = "eda"
    gstate["intent"] = intent
    return gstate


def route_by_intent(gstate: GraphQAState) -> str:
    return gstate.get("intent", "eda")


def node_answer(agent, gstate: GraphQAState, fallback: str) -> GraphQAState:
    ans = agent.answer(gstate["app_state"], gstate["question"])
    gstate["final_answer"] = ans or fallback
    return gstate


def feature_node(gstate): return node_answer(FeatureExplainerAgent(), gstate, "I could not identify the KPI. Try `can you give me RoofArea`.")
def raw_node(gstate): return node_answer(RawDataQAAgent(), gstate, "I could not parse raw-data question.")
def eda_node(gstate): return node_answer(EDAQAAgent(), gstate, "Try `Show EDA pattern summary` or `What are the business rules?`.")
def monotonic_node(gstate): return node_answer(MonotonicQAAgent(), gstate, "Try `Show monotonic violation summary`.")
def target_order_node(gstate): return node_answer(TargetOrderedBreakAgent(), gstate, "Try `Show target ordered monotonic break analysis`.")
def kpi_to_target_node(gstate): return node_answer(KPIToTargetAgent(), gstate, "Try `If RelativeCompactness is between 0.7 and 0.79, what is expected HeatingLoad?`.")
def target_to_kpi_node(gstate): return node_answer(TargetToKPIAgent(), gstate, "Try `If HeatingLoad target is between 15 and 20, tell me KPI ranges`.")
def expert_node(gstate): return node_answer(ExpertReasoningAgent(), gstate, "Try `Why is HeatingLoad high when RelativeCompactness is between 0.7 and 0.79?`.")
def repair_node(gstate): return node_answer(RepairAgent(), gstate, "Try `Repair RelativeCompactness value 0.76 to nearest valid value`.")
def scenario_node(gstate): return node_answer(ScenarioSimulationRepairAgent(), gstate, "Try `Simulate RelativeCompactness=0.76 RoofArea=150 and repair invalid KPIs`.")


@st.cache_resource(show_spinner=False)
def build_langgraph_app():
    graph = StateGraph(GraphQAState)
    graph.add_node("classify", classify_intent_node)
    for name, fn in [
        ("feature", feature_node),
        ("raw", raw_node),
        ("eda", eda_node),
        ("monotonic", monotonic_node),
        ("target_order", target_order_node),
        ("kpi_to_target", kpi_to_target_node),
        ("target_to_kpi", target_to_kpi_node),
        ("expert", expert_node),
        ("repair", repair_node),
        ("scenario", scenario_node),
    ]:
        graph.add_node(name, fn)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_by_intent, {
        "feature": "feature",
        "raw": "raw",
        "eda": "eda",
        "monotonic": "monotonic",
        "target_order": "target_order",
        "kpi_to_target": "kpi_to_target",
        "target_to_kpi": "target_to_kpi",
        "expert": "expert",
        "repair": "repair",
        "scenario": "scenario",
    })
    for node in ["feature", "raw", "eda", "monotonic", "target_order", "kpi_to_target", "target_to_kpi", "expert", "repair", "scenario"]:
        graph.add_edge(node, END)
    return graph.compile()


# =========================================================
# PIPELINE ORCHESTRATOR
# =========================================================

class A2AOrchestrator:
    def __init__(self, target_col: str):
        self.target_col = target_col

    def run(self) -> AgentState:
        state = AgentState(target_col=self.target_col)
        agents = [
            DataLoaderAgent(self.target_col),
            DataQualityAgent(),
            FeatureEDAAgent(),
            EDAPatternAgent(),
            EDAMonotonicityAgent(),
            ModelTrainingAgent(),
            PDPMonotonicityAgent(),
            SHAPValidationAgent(),
            RuleZoneAgent(),
            KnowledgeGraphAgent(),
        ]
        for agent in agents:
            state = agent.run(state)
        return state


# =========================================================
# UI HELPERS
# =========================================================

def get_top_shap_features(state: AgentState, top_n: int = 5) -> List[str]:
    features = list(state.rules.keys())
    if not state.shap_report.get("available"):
        return features[:top_n]
    data = state.shap_report.get("features", {})
    if not data:
        return features[:top_n]
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "feature"})
    df = df.sort_values("mean_abs_shap", ascending=False)
    return df["feature"].head(top_n).tolist()


def get_shap_feature_table(state: AgentState) -> pd.DataFrame:
    if not state.shap_report.get("available"):
        return pd.DataFrame({"feature": list(state.rules.keys()), "mean_abs_shap": np.nan, "status": "SHAP unavailable"})
    return pd.DataFrame(state.shap_report.get("features", {})).T.reset_index().rename(columns={"index": "feature"}).sort_values("mean_abs_shap", ascending=False)


def plot_scatter(df, feature, target):
    fig, ax = plt.subplots()
    ax.scatter(df[feature], df[target], alpha=0.6)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}")
    return fig


def plot_bin_trend(state, feature):
    mono = state.monotonic_report["eda_monotonicity"][feature]
    bin_df = pd.DataFrame(mono["bin_table"])
    fig, ax = plt.subplots()
    ax.plot(range(len(bin_df)), bin_df["target_mean"], marker="o")
    ax.set_xlabel("Ordered KPI bins")
    ax.set_ylabel(f"Mean {state.target_col}")
    ax.set_title(f"Monotonic Trend: {feature} ({mono['rule']})")
    return fig


def plot_correlation_bar(state):
    corr = state.eda_report["correlations"]
    df = pd.DataFrame({"feature": list(corr.keys()), "correlation": list(corr.values())})
    fig, ax = plt.subplots()
    ax.bar(df["feature"], df["correlation"])
    ax.set_ylabel(f"Correlation with {state.target_col}")
    ax.set_title("Feature Correlation")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_safe_violation_ranges(state, feature):
    selected = state.monotonic_report["range_summary"][feature]
    fig, ax = plt.subplots()
    for low, high in selected.get("safe_ranges", []): ax.hlines(1, low, high, linewidth=8)
    for low, high in selected.get("violation_ranges", []): ax.hlines(0, low, high, linewidth=8)
    for low, high in selected.get("info_ranges", []): ax.hlines(0.5, low, high, linewidth=8)
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["Violation", "Info", "Safe"])
    ax.set_xlabel(feature); ax.set_title(f"Safe vs Violation Ranges: {feature}")
    return fig


# =========================================================
# STREAMLIT APP
# =========================================================

def main():
    st.set_page_config(page_title="A2A V8 Expert System", layout="wide")
    st.title("A2A V8: Gemini + LangGraph Expert System")
    st.caption("Target-ordered monotonicity, confounders, scenario simulation, global/local repair, and feature-aware routing")

    with st.sidebar:
        target_col = st.selectbox("Target", TARGET_OPTIONS, index=0)
        gemini_model = st.text_input("Gemini model", DEFAULT_GEMINI_MODEL)
        api_key_input = st.text_input("Gemini API key optional", type="password")
        use_gemini_polish = st.checkbox("Use Gemini to polish deterministic answer", value=False)
        with st.expander("Sample validation questions"):
            for q in SAMPLE_QUESTIONS:
                st.code(q)
        if st.button("Run / Refresh Pipeline"):
            with st.spinner("Running deterministic expert pipeline..."):
                st.session_state.app_state = A2AOrchestrator(target_col).run()
                st.session_state.messages = []
                st.session_state.gemini = GeminiClient(gemini_model, api_key_input or None)
                st.session_state.langgraph_app = build_langgraph_app()
            st.success("Pipeline ready")

    if "app_state" not in st.session_state:
        st.info("Click **Run / Refresh Pipeline** first.")
        st.code("pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit networkx openpyxl google-genai langgraph typing_extensions")
        return

    state: AgentState = st.session_state.app_state
    if "langgraph_app" not in st.session_state:
        st.session_state.langgraph_app = build_langgraph_app()
    if "gemini" not in st.session_state:
        st.session_state.gemini = GeminiClient()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chat", "Target-Ordered Breaks", "Scenario Simulation", "EDA Patterns", "Monotonicity & Ranges", "Plots / Data"])

    with tab1:
        st.subheader("LangGraph Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask expert system question...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                result = st.session_state.langgraph_app.invoke({"question": question, "app_state": state})
                answer = result.get("final_answer", "No answer generated.")
                if use_gemini_polish:
                    answer = gemini_polish_answer(st.session_state.gemini, question, answer)
                st.markdown(answer)
                with st.expander("Routing info"):
                    st.json({"intent": result.get("intent")})
            st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab2:
        st.subheader("Target-Ordered Monotonic Break Analysis")
        feature_choice = st.selectbox("Analyze feature", ["ALL"] + list(state.rules.keys()), key="target_order_feature")
        requested = None if feature_choice == "ALL" else feature_choice
        result = TargetOrderedBreakAgent().analyze(state, requested)
        st.markdown("### KPI Summary")
        st.dataframe(pd.DataFrame(result["feature_summaries"]), use_container_width=True)
        st.markdown("### Target Ordered Table")
        st.dataframe(pd.DataFrame(result["target_order_table"]), use_container_width=True)
        st.markdown("### Break Cases")
        breaks = pd.DataFrame(result["break_cases"])
        if breaks.empty:
            st.success("No break cases found.")
        else:
            st.dataframe(breaks.drop(columns=["confounders"], errors="ignore"), use_container_width=True)
            idx = st.number_input("Inspect break case index", min_value=0, max_value=max(len(result["break_cases"]) - 1, 0), value=0, step=1)
            st.json(result["break_cases"][int(idx)])

    with tab3:
        st.subheader("ML Scenario Simulation + Global/Local Repair")
        st.caption("Provide KPI values. Missing KPIs use median baseline. Invalid unidirectional KPI values are repaired to nearest valid raw training value.")
        cols = st.columns(4)
        assignments = {}
        for i, feature in enumerate(state.rules.keys()):
            with cols[i % 4]:
                use_feature = st.checkbox(feature, value=False, key=f"sim_use_{feature}")
                if use_feature:
                    default = float(state.df[feature].median())
                    assignments[feature] = st.number_input(feature, value=default, step=0.01, key=f"sim_val_{feature}")
        if st.button("Run Simulation + Repair"):
            if not assignments:
                st.warning("Select at least one KPI.")
            else:
                sim = ScenarioSimulationRepairAgent().simulate(state, assignments)
                st.markdown(f"### Prediction before repair: **{sim['prediction_before_repair']:.3f}**")
                st.markdown(f"### Prediction after repair: **{sim['prediction_after_repair']:.3f}**")
                st.subheader("Repairs")
                st.dataframe(pd.DataFrame(sim["repairs"]), use_container_width=True)
                st.subheader("Scenario before repair")
                st.json(sim["scenario_before_repair"])
                st.subheader("Scenario after repair")
                st.json(sim["scenario_after_repair"])
                st.subheader("Local check after repair")
                st.dataframe(pd.DataFrame(sim["local_check_after_repair"]["local_checks"]), use_container_width=True)

    with tab4:
        st.subheader("EDA Pattern Agent")
        st.markdown(EDAQAAgent().answer(state, "Show EDA pattern summary"))
        pattern = state.eda_report.get("pattern_agent", {})
        rows = []
        for f, d in pattern.get("univariate_patterns", {}).items():
            rows.append({"feature": f, "rule": d["rule"], "correlation_with_target": d["correlation_with_target"], "alignment": d["business_rule_alignment"], "recommendation": d["recommendation"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.subheader("Top Interactions")
        st.dataframe(pd.DataFrame(pattern.get("top_multivariate_patterns", [])), use_container_width=True)

    with tab5:
        st.subheader("Safe / Violation Ranges")
        summary = state.monotonic_report.get("range_summary", {})
        st.dataframe(ZoneRangeExtractor.to_dataframe(summary), use_container_width=True)
        f = st.selectbox("Feature", list(summary.keys()), key="range_feature")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Safe")
            st.dataframe(pd.DataFrame(summary[f].get("safe_ranges", []), columns=["low", "high"]), use_container_width=True)
        with c2:
            st.markdown("### Violation")
            st.dataframe(pd.DataFrame(summary[f].get("violation_ranges", []), columns=["low", "high"]), use_container_width=True)
        st.subheader("SHAP")
        st.dataframe(get_shap_feature_table(state), use_container_width=True)

    with tab6:
        st.subheader("Plots / Data")
        plot_features = get_top_shap_features(state, min(5, len(state.rules)))
        f = st.selectbox("Plot feature", plot_features, key="plot_feature")
        st.pyplot(plot_scatter(state.df, f, state.target_col))
        st.pyplot(plot_bin_trend(state, f))
        st.pyplot(plot_safe_violation_ranges(state, f))
        st.pyplot(plot_correlation_bar(state))
        st.subheader("Model Report")
        st.json(state.model_report)
        st.subheader("Raw Data")
        st.dataframe(state.df, use_container_width=True)


if __name__ == "__main__":
    main()
