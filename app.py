import gradio as gr
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os
from io import BytesIO
from matplotlib.patches import Wedge

MODELS = {
    "XGB Small": {
        "model_path": "xgb_small.pkl",
        "numeric_cols": ['Age','SurfaceA','MeanRadius','AspectRatio','SizeRatio'],
        "categorical_cols": ['Type','Location'],
        "type": "xgb"
    },
    "XGB Hemodynamic": {
        "model_path": "xgb_hemo.pkl",
        "numeric_cols": [
            'Age','Volume','ShapeFactor','Torsion','MeanRadius','AspectRatio',
            'SizeRatio','TAWSS','PressureDrop','OSI','MeanVelocity','LSA',
            'LSApercent','transWSS','RRT','EL','LNH'
        ],
        "categorical_cols": ['Sex','Type','Location','MultipleAneyrsm'],
        "type": "xgb"
    },
    "XGB Big": {
        "model_path": "xgb_big.pkl",
        "numeric_cols": ['Age', 'SurfaceA', 'Volume', 'ShapeFactor', 'MeanCurvature', 'MeanRadius'],
        "categorical_cols": ['Sex','Location', 'MultipleAneurysm'],
        "type": "xgb"
    },
    "MLP Small": {
        "model_path": "mlp_small.pkl",
        "numeric_cols": ['Age','SurfaceA','Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius','AspectRatio','SizeRatio'],
        "categorical_cols": ['Sex', 'Type','Location', 'MultipleAneurysm'],
        "type": "mlp"
    },
    "MLP Hemodynamic": {
        "model_path": "mlp_hemo.pkl",
        "numeric_cols": ['Age','SurfaceA','Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius','AspectRatio','SizeRatio', 'TAWSS', 'PressureDrop', 'OSI', 'MeanVelocity', 'LSA', 'LSApercent', 'transWSS', 'RRT', 'WSR', 'EL', 'LNH', 'VortexNumber'],
        "categorical_cols": ['Sex', 'Type','Location', 'MultipleAneyrsm'],
        "type": "mlp"
    },
    "MLP Big": {
        "model_path": "mlp_big.pkl",
        "numeric_cols": ['Age','SurfaceA','Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius','AspectRatio','SizeRatio'],
        "categorical_cols": ['Sex', 'Type','Location', 'MultipleAneurysm'],
        "type": "mlp"
    },
    "RF Small": {
        "model_path": "rf_small_model.pkl",
        "numeric_cols": ['Age', 'SurfaceA', 'Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius', 'AspectRatio', 'SizeRatio'],
        "categorical_cols": ['Sex', 'Type', 'Location', 'MultipleAneurysm'],
        "type": "rf"
    },
    "RF Big": {
        "model_path": "rf_big_model.pkl",
        "numeric_cols": ['Age', 'SurfaceA', 'Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius', 'AspectRatio', 'SizeRatio'],
        "categorical_cols": ['Sex', 'Type', 'Location', 'MultipleAneurysm'],
        "type": "rf"
    },
    "RF Hemodynamic": {
        "model_path": "rf_hemo_all_features.pkl",
        "numeric_cols": ['Age', 'SurfaceA', 'Volume', 'ShapeFactor', 'MeanCurvature', 'Torsion', 'MeanRadius', 'AspectRatio', 'SizeRatio', 'TAWSS', 'PressureDrop', 'OSI', 'MeanVelocity', 'LSA', 'LSApercent', 'transWSS', 'RRT', 'WSR', 'EL', 'LNH', 'VortexNumber'],
        "categorical_cols": ['Sex', 'Type', 'Location', 'MultipleAneyrsm'],
        "type": "rf"
    }
}

def pil_image_dims(buf):
    from PIL import Image as PILImage
    buf.seek(0)
    im = PILImage.open(buf)
    w, h = im.size
    buf.seek(0)
    return w, h

def image_with_aspect(buf, max_width, max_height):
    w, h = pil_image_dims(buf)
    ratio = min(max_width / w, max_height / h)
    return Image(buf, width=w * ratio, height=h * ratio)

def circular_probability_gauge(prob, pred_label, font_family="DejaVu Sans"):
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(2.5,2.5), subplot_kw={'aspect': 'equal'})
    circle_bg = plt.Circle((0,0), 1, color='#e6e6e6', zorder=1)
    ax.add_patch(circle_bg)
    arc = Wedge(center=(0,0), r=1, theta1=0, theta2=360*prob, width=0.15, color='#0052cc', alpha=0.9, zorder=2)
    ax.add_patch(arc)
    ax.text(0,0.2, f"{prob:.1%}", va="center", ha="center", fontsize=26, fontweight='bold', fontname=font_family)
    ax.text(0,-0.3, pred_label, va="center", ha="center", fontsize=15, fontname=font_family)
    ax.axis("off")
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return buf

def plot_age_pdp(pipe, df_new, m, all_cols, proba):
    buf = BytesIO()
    age_val = float(df_new["Age"].values[0])
    age_range = np.arange(age_val, min(90, age_val + 25) + 1, 1)
    proba_vals = []
    for a in age_range:
        row = df_new.copy()
        row["Age"] = a
        if m["type"] == "rf":
            proba_a = pipe.predict_proba(row)[0, 1]
        elif m["type"] == "xgb":
            X_new = pipe.named_steps['pre'].transform(row)
            proba_a = pipe.named_steps['clf'].predict_proba(X_new)[0, 1]
        elif m["type"] == "mlp":
            proba_a = pipe.predict_proba(row)[0, 1]
        proba_vals.append(proba_a)
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    ax.plot(age_range, proba_vals, color='blue', lw=2)
    ax.scatter([age_val], [proba], color='red', s=40, zorder=5, label="Patient Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Predicted Rupture Probability")
    ax.set_title("Partial Dependence (Age)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_text_explanation(exp, df_new):
    abs_vals = np.abs(exp.values)
    order = np.argsort(abs_vals)[::-1]
    top_feats = np.array(exp.feature_names)[order][:2]
    top_vals = np.array(exp.data)[order][:2]
    text = "This risk is mainly influenced by: "
    pieces = []
    for feat, val in zip(top_feats, top_vals):
        try:
            v = float(val)
        except Exception:
            v = val
        if isinstance(v, (int, float)):
            pieces.append(f"{feat} = {v:.2f}")
        else:
            pieces.append(f"{feat} = {v}")
    text += ", ".join(pieces)
    return text

def patient_report_pdf(df_input, model_key):
    m = MODELS[model_key]
    pipe = joblib.load(m["model_path"])
    all_cols = m["numeric_cols"] + m["categorical_cols"]

    tmp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(tmp_dir, "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    df_nn = df_input[all_cols].copy()

    for idx, patient in df_input.iterrows():
        patient_data = patient[all_cols].to_dict()
        for k in m["categorical_cols"]:
            patient_data[k] = int(patient_data[k])
        df_new = pd.DataFrame([patient_data], columns=all_cols)

        if m["type"] == "rf":
            expected_features = list(pipe.feature_names_in_)
            df_new = df_new[expected_features]
            df_nn_used = df_nn[expected_features].copy()
            all_cols_used = expected_features
        else:
            df_nn_used = df_nn.copy()
            all_cols_used = all_cols

        pred = pipe.predict(df_new)[0]
        proba = pipe.predict_proba(df_new)[0, 1]
        pred_label = 'Ruptured' if pred == 1 else 'Unruptured'

        elements.append(Paragraph(f"<b>Patient {idx+1} Profile ({model_key})</b>", styles['Title']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Predicted class:</b> {pred_label}", styles['Normal']))
        elements.append(Paragraph(f"<b>Probability:</b> {proba:.3f}", styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in patient_data.items()]
        tbl = Table(table_data, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 14))

        waterfall_buf = BytesIO()
        bar_buf = BytesIO()
        try:
            if m["type"] == "xgb":
                max_bg = min(200, len(df_nn_used))
                df_bg = df_nn_used.sample(n=max_bg, replace=(max_bg > len(df_nn_used)), random_state=42)
                X_bg = pipe.named_steps['pre'].transform(df_bg)
                X_new = pipe.named_steps['pre'].transform(df_new)
                clf = pipe.named_steps['clf']
                try:
                    feat_names = pipe.named_steps['pre'].get_feature_names_out()
                except Exception:
                    feat_names = [str(i) for i in range(X_new.shape[1])]
                explainer = shap.TreeExplainer(clf, data=X_bg, feature_names=feat_names, model_output="probability")
                shap_vals = explainer.shap_values(X_new)
                exp = shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=X_new[0],
                    feature_names=feat_names
                )
            elif m["type"] == "mlp":
                n_bg = min(200, len(df_nn_used))
                df_bg = df_nn_used.sample(n=n_bg, random_state=42)
                bg_vals = df_bg.values
                def f(x): return pipe.predict_proba(pd.DataFrame(x, columns=all_cols_used))[:,1]
                explainer = shap.KernelExplainer(f, bg_vals, link="identity")
                shap_vals = explainer.shap_values(df_new.values, nsamples=100)
                exp = shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=df_new.values[0],
                    feature_names=all_cols_used
                )
            elif m["type"] == "rf":
                explainer = shap.TreeExplainer(pipe)
                shap_vals = explainer.shap_values(df_new)
                exp = shap.Explanation(
                    values=shap_vals[1][0],
                    base_values=explainer.expected_value[1],
                    data=df_new.iloc[0].values,
                    feature_names=all_cols_used
                )

            # LARGER SHAP Waterfall
            plt.figure(figsize=(10.5, 5.1))  # was (7, 3)
            shap.plots.waterfall(exp, max_display=7, show=False)
            plt.tight_layout()
            plt.savefig(waterfall_buf, format="png", bbox_inches="tight", dpi=170, facecolor="white")
            plt.close()
            waterfall_buf.seek(0)

            # LARGER Feature Contribution Magnitudes
            abs_vals = np.abs(exp.values)
            order = np.argsort(abs_vals)[::-1]
            plt.figure(figsize=(9.5, 3.7))  # was (8, 4)
            plt.barh(np.array(exp.feature_names)[order][:10], abs_vals[order][:10], color='tab:blue')
            plt.xlabel("Absolute SHAP value (probability)")
            plt.title("Feature Contribution Magnitudes")
            plt.tight_layout()
            plt.savefig(bar_buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
            plt.close()
            bar_buf.seek(0)
        except Exception as e:
            plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No SHAP explanation", ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(waterfall_buf, format="png", bbox_inches="tight", dpi=120, facecolor="white")
            plt.close()
            waterfall_buf.seek(0)
            plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No feature importances", ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(bar_buf, format="png", bbox_inches="tight", dpi=120, facecolor="white")
            plt.close()
            bar_buf.seek(0)

        elements.append(Paragraph("SHAP Waterfall", styles['Heading4']))
        elements.append(image_with_aspect(waterfall_buf, max_width=520, max_height=250))  # larger
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Feature Importances", styles['Heading4']))
        elements.append(image_with_aspect(bar_buf, max_width=440, max_height=160))  # larger
        elements.append(Spacer(1, 10))

        gauge_buf = circular_probability_gauge(proba, pred_label, font_family="DejaVu Sans")
        elements.append(Paragraph("Predicted Probability Gauge", styles['Heading4']))
        elements.append(image_with_aspect(gauge_buf, max_width=230, max_height=115))
        elements.append(Spacer(1, 12))

        age_pdp_buf = plot_age_pdp(pipe, df_new, m, all_cols_used, proba)
        elements.append(Paragraph("Partial Dependence Plot: Age", styles['Heading4']))
        elements.append(image_with_aspect(age_pdp_buf, max_width=340, max_height=155))
        elements.append(Spacer(1, 10))

        try:
            text_expl = generate_text_explanation(exp, df_new)
        except Exception:
            text_expl = "No textual explanation available."
        elements.append(Paragraph(f"<b>Explanation:</b> {text_expl}", styles['Normal']))
        elements.append(PageBreak())

    doc.build(elements)
    return pdf_path

def batch_predict_pdf(file, model_key):
    if file is None or model_key not in MODELS:
        return None
    df_input = pd.read_excel(file.name)
    model = MODELS[model_key]
    req_numeric = model["numeric_cols"]
    req_categorical = model["categorical_cols"]
    col_map = {}
    req_cols = req_numeric + req_categorical
    if "MultipleAneurysm" in req_cols or "MultipleAneyrsm" in req_cols:
        if "MultipleAneurysm" in df_input.columns and "MultipleAneyrsm" not in df_input.columns:
            col_map["MultipleAneurysm"] = "MultipleAneurysm"
        elif "MultipleAneyrsm" in df_input.columns and "MultipleAneurysm" not in df_input.columns:
            col_map["MultipleAneyrsm"] = "MultipleAneurysm" if "MultipleAneurysm" in req_cols else "MultipleAneyrsm"
        elif "MultipleAneurysm" in df_input.columns and "MultipleAneyrsm" in df_input.columns:
            if "MultipleAneurysm" in req_cols:
                col_map["MultipleAneurysm"] = "MultipleAneurysm"
            else:
                col_map["MultipleAneyrsm"] = "MultipleAneyrsm"
    df_input = df_input.rename(columns=col_map)
    required_cols = set(req_numeric + req_categorical)
    for alt_col in ["MultipleAneurysm", "MultipleAneyrsm"]:
        if alt_col in required_cols:
            if (alt_col not in df_input.columns) and (("MultipleAneurysm" in df_input.columns) or ("MultipleAneyrsm" in df_input.columns)):
                required_cols.discard(alt_col)
    missing = required_cols - set(df_input.columns)
    if missing:
        raise gr.Error(f"Missing columns in input file: {', '.join(missing)}")
    pdf_path = patient_report_pdf(df_input, model_key)
    return pdf_path

with gr.Blocks() as demo:
    gr.Markdown("# Aneurysm Rupture Predictor – Batch Report Generator")
    gr.Markdown(
        "Upload an Excel file (.xlsx) of patient data, choose the model, and download a PDF report with each patient's prediction, explainability, and profile."
    )
    with gr.Row():
        file_input = gr.File(label="Upload Excel (.xlsx) file")
        model_choice = gr.Dropdown(list(MODELS.keys()), label="Choose model")
        run_btn = gr.Button("Generate Patient Reports (PDF)")
    pdf_output = gr.File(label="Download report.pdf")

    run_btn.click(
        fn=batch_predict_pdf,
        inputs=[file_input, model_choice],
        outputs=pdf_output
    )

    gr.Markdown("---")
    with gr.Column():
        gr.Markdown("## Required columns for each model")
        for key, model in MODELS.items():
            cols = model["numeric_cols"] + model["categorical_cols"]
            cols_str = "<br>".join(f"<code>{c}</code>" for c in cols)
            tip = ""
            if "MultipleAneurysm" in cols or "MultipleAneyrsm" in cols:
                tip = "<br><small>Tip: Both <code>MultipleAneurysm</code> and <code>MultipleAneyrsm</code> accepted for input.</small>"
            with gr.Accordion(label=f"Required columns for {key}", open=False):
                gr.Markdown(cols_str + tip)

if __name__ == "__main__":
    demo.launch()
