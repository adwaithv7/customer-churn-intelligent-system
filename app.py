"""
Customer Churn Intelligence System - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_auc_score, roc_curve, accuracy_score
)

st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="📊",
    layout="wide"
)

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10
})

# Loaded and trained the dataset
@st.cache_data
def load_and_train():
    df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Preprocessing
    df = df_raw.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv_f1   = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1', n_jobs=-1)
        results[name] = {
            'model':    model,
            'y_pred':   y_pred,
            'y_proba':  y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1':       f1_score(y_test, y_pred),
            'roc_auc':  roc_auc_score(y_test, y_proba),
            'cv_mean':  cv_f1.mean(),
            'cv_std':   cv_f1.std(),
        }

    best_name = max(results, key=lambda k: results[k]['f1'])

    # Risk segmentation on full dataset
    best_proba  = results[best_name]['model'].predict_proba(X_scaled)[:, 1]
    risk_labels = pd.cut(best_proba, bins=[0, 0.35, 0.65, 1.0],
                         labels=['Low', 'Medium', 'High'])

    risk_df = df_raw[['tenure', 'Contract', 'MonthlyCharges',
                       'InternetService', 'PaymentMethod']].copy()
    risk_df['Churn_Probability (%)'] = (best_proba * 100).round(1)
    risk_df['Risk_Level'] = risk_labels.tolist()

    return df_raw, X, y, X_scaled, X_test, y_test, results, best_name, risk_df


with st.spinner("Training models — please wait..."):
    df_raw, X, y, X_scaled, X_test, y_test, results, best_name, risk_df = load_and_train()

best = results[best_name]

# Sidebar
st.sidebar.title("Customer Churn Intelligence")
st.sidebar.caption("Adwaith V ML Project(GUVI HCL)")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "EDA",
    "Model Performance",
    "Risk Segmentation",
    "Business Insights",
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best model:** {best_name}")
st.sidebar.markdown(f"**F1-Score:** {best['f1']:.3f}")
st.sidebar.markdown(f"**ROC-AUC:** {best['roc_auc']:.3f}")


#Pg1 - overview
if page == "Overview":
    st.title("Customer Churn Intelligence System")
    st.caption("Telecom dataset · 7,043 customers · 19 features")
    st.markdown("---")

    total     = len(risk_df)
    high      = (risk_df['Risk_Level'] == 'High').sum()
    medium    = (risk_df['Risk_Level'] == 'Medium').sum()
    low       = (risk_df['Risk_Level'] == 'Low').sum()
    churn_pct = (df_raw['Churn'] == 'Yes').mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churn Rate",      f"{churn_pct:.1f}%")
    c3.metric("High Risk",       f"{high:,}")
    c4.metric("Medium Risk",     f"{medium:,}")
    c5.metric("Low Risk",        f"{low:,}")

    st.markdown("---")
    st.subheader("Model Comparison")

    summary = pd.DataFrame([{
        'Model':      name,
        'Accuracy':   f"{res['accuracy']:.3f}",
        'F1-Score':   f"{res['f1']:.3f}",
        'ROC-AUC':    f"{res['roc_auc']:.3f}",
        'CV F1 Mean': f"{res['cv_mean']:.3f}",
        'CV F1 Std':  f"±{res['cv_std']:.3f}",
        'Best':       '✅' if name == best_name else ''
    } for name, res in results.items()])
    st.dataframe(summary, use_container_width=True, hide_index=True)


#Pg2 - data
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    df_eda = df_raw.copy()
    df_eda['TotalCharges'] = pd.to_numeric(df_eda['TotalCharges'], errors='coerce')
    df_eda['Churn_bin']    = (df_eda['Churn'] == 'Yes').astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Churn Patterns — Key Variables', fontsize=13, fontweight='bold')

    
    c_churn = df_eda.groupby('Contract')['Churn_bin'].mean() * 100
    axes[0, 0].bar(c_churn.index, c_churn.values, color='steelblue', width=0.5)
    axes[0, 0].set_title('Churn Rate by Contract Type')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(c_churn.values):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    
    axes[0, 1].hist(df_eda[df_eda['Churn']=='No']['tenure'],  bins=30, alpha=0.6,
                    label='No Churn', color='steelblue')
    axes[0, 1].hist(df_eda[df_eda['Churn']=='Yes']['tenure'], bins=30, alpha=0.6,
                    label='Churn',    color='tomato')
    axes[0, 1].set_title('Tenure Distribution by Churn')
    axes[0, 1].set_xlabel('Tenure (months)')
    axes[0, 1].legend()

    
    axes[0, 2].boxplot(
        [df_eda[df_eda['Churn']=='No']['MonthlyCharges'],
         df_eda[df_eda['Churn']=='Yes']['MonthlyCharges']],
        labels=['No Churn', 'Churn'], patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.5),
        medianprops=dict(color='tomato', linewidth=2)
    )
    axes[0, 2].set_title('Monthly Charges vs Churn')
    axes[0, 2].set_ylabel('Monthly Charges ($)')

    
    i_churn = df_eda.groupby('InternetService')['Churn_bin'].mean() * 100
    axes[1, 0].bar(i_churn.index, i_churn.values, color='steelblue', width=0.5)
    axes[1, 0].set_title('Churn Rate by Internet Service')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(i_churn.values):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    
    p_churn = df_eda.groupby('PaymentMethod')['Churn_bin'].mean() * 100
    short   = [p.split('(')[0].strip() for p in p_churn.index]
    axes[1, 1].barh(short, p_churn.values, color='steelblue')
    axes[1, 1].set_title('Churn Rate by Payment Method')
    axes[1, 1].set_xlabel('Churn Rate (%)')

    
    s_churn = df_eda.groupby('SeniorCitizen')['Churn_bin'].mean() * 100
    axes[1, 2].bar(['Non-Senior', 'Senior'], s_churn.values, color='steelblue', width=0.4)
    axes[1, 2].set_title('Churn Rate by Senior Citizen')
    axes[1, 2].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(s_churn.values):
        axes[1, 2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


#Pg3 - model perfomance
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(f'{name}\nF1={res["f1"]:.3f} | AUC={res["roc_auc"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ROC curves
    with col1:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={res["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Models')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # CV scores
    with col2:
        st.subheader("5-Fold Cross-Validation F1")
        fig, ax = plt.subplots(figsize=(7, 5))
        names    = list(results.keys())
        cv_means = [results[n]['cv_mean'] for n in names]
        cv_stds  = [results[n]['cv_std']  for n in names]
        bars = ax.barh(names, cv_means, xerr=cv_stds,
                       color='steelblue', alpha=0.8, capsize=5, height=0.4)
        ax.set_xlabel('F1-Score (mean ± std)')
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, cv_means):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Classification reports
    st.subheader("Classification Reports")
    for name, res in results.items():
        with st.expander(f"{name}"):
            st.code(classification_report(y_test, res['y_pred'],
                                          target_names=['No Churn', 'Churn']))

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance & Interpretability")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    rf_imp = pd.Series(results['Random Forest']['model'].feature_importances_,
                        index=X.columns).sort_values()
    rf_imp.plot(kind='barh', ax=axes[0], color='steelblue', alpha=0.8)
    axes[0].set_title('Random Forest — Feature Importance')
    axes[0].set_xlabel('Importance Score')

    lr_coef = pd.Series(np.abs(results['Logistic Regression']['model'].coef_[0]),
                         index=X.columns).sort_values()
    lr_coef.plot(kind='barh', ax=axes[1], color='steelblue', alpha=0.8)
    axes[1].set_title('Logistic Regression — |Coefficient|')
    axes[1].set_xlabel('Absolute Coefficient Value')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


#Pg4 - risk segmentation
elif page == "Risk Segmentation":
    st.title("Churn Risk Segmentation")
    st.caption(f"Risk scores assigned by: {best_name}")
    st.markdown("---")

    st.subheader("Segment Summary")
    seg_summary = risk_df.groupby('Risk_Level').agg(
        Count              =('Risk_Level',           'count'),
        Avg_Churn_Prob     =('Churn_Probability (%)', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges',        'mean'),
        Avg_Tenure         =('tenure',                'mean')
    ).round(1).reindex(['High', 'Medium', 'Low'])
    st.dataframe(seg_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Risk Table")

    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect("Filter by Risk Level",
                                     ['High', 'Medium', 'Low'],
                                     default=['High', 'Medium', 'Low'])
    with col2:
        min_prob = st.slider("Min Churn Probability (%)", 0, 100, 0)

    filtered = risk_df[
        risk_df['Risk_Level'].isin(risk_filter) &
        (risk_df['Churn_Probability (%)'] >= min_prob)
    ].sort_values('Churn_Probability (%)', ascending=False)

    st.caption(f"Showing {len(filtered):,} customers")
    st.dataframe(filtered.head(300), use_container_width=True, hide_index=True)


#Pg5 - business insights
elif page == "Business Insights":
    st.title("Business Insights & Retention Strategies")
    st.markdown("---")

    st.subheader("Key Findings")
    findings = pd.DataFrame([
        ("Contract type",    "Month-to-month churn ~43% vs <3% for 2-year",  "Incentivize long-term contracts with discounts"),
        ("Tenure",           "Customers < 12 months are most at risk",        "Proactive check-in calls at months 3, 6, 9"),
        ("Internet service", "Fiber optic churn ~42%",                        "Investigate service quality, offer SLA guarantee"),
        ("Payment method",   "Electronic check churn ~45%",                   "Incentivize auto-pay switch with 10% discount"),
        ("Support services", "No TechSupport → high churn",                   "Bundle support into base plan or free trial"),
        ("Senior citizens",  "Senior + no partner → ~35% churn",             "Assign dedicated support representative"),
    ], columns=["Driver", "Finding", "Action"])
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Retention Strategy by Risk Tier")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**High Risk (>65%)**")
        st.markdown("""
- Immediate personal outreach
- Offer contract upgrade with 20% discount
- Escalate unresolved support tickets
- Assign dedicated account manager
        """)
    with col2:
        st.markdown("**Medium Risk (35–65%)**")
        st.markdown("""
- Personalized value-reminder email
- Loyalty discount or service bundle
- Satisfaction survey and follow-up
- Highlight savings vs competitors
        """)
    with col3:
        st.markdown("**Low Risk (<35%)**")
        st.markdown("""
- Cross-sell streaming, backup, device protection
- Launch referral reward program
- Ensure smooth billing experience
- Periodic satisfaction check-ins
        """)

    st.markdown("---")
    st.subheader("Final Model Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Dataset | 7,043 customers, 19 features |
| Churn rate | 26.5% |
| Best model | {best_name} |
| Test F1-Score | {best['f1']:.3f} |
| ROC-AUC | {best['roc_auc']:.3f} |
| CV F1 (5-fold) | {best['cv_mean']:.3f} ± {best['cv_std']:.3f} |
| Test Accuracy | {best['accuracy']:.3f} |
        """)
    with col2:
        for level in ['High', 'Medium', 'Low']:
            count = (risk_df['Risk_Level'] == level).sum()
            pct   = count / len(risk_df) * 100
            st.metric(f"{level} Risk", f"{count:,} customers", f"{pct:.1f}%")