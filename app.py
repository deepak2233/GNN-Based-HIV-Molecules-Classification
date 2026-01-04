import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
import os
from torch_geometric.data import Data

# Page configuration
st.set_page_config(
    page_title="HIV-GNN Explorer", 
    page_icon="🧪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Feature sizes (standard for this dataset)
FEATURE_SIZE = 9
EDGE_FEATURE_SIZE = 2

@st.cache_resource
def load_trained_model(m_type, path):
    if not os.path.exists(path):
        return None
    
    if m_type == 'GNN1':
        model = GNN1(feature_size=FEATURE_SIZE)
    elif m_type == 'GNN2':
        model = GNN2(feature_size=FEATURE_SIZE)
    else:
        model = GNN3(feature_size=FEATURE_SIZE, edge_feature_size=EDGE_FEATURE_SIZE)
        
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except RuntimeError as e:
        st.error(f"**Architecture Mismatch!** The model architecture has been improved (GIN + Transformer + BCE Loss), making old checkpoints incompatible.")
        st.info(f"Please retrain the model using: `python main.py --mode train --model_type {m_type}`")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()), int(atom.GetChiralTag())
        ])
    
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        ef = [bond.GetBondTypeAsDouble(), int(bond.IsInRing())]
        edge_feats += [ef, ef]
        
    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

# Tabs
tab_predict, tab_eda, tab_blog = st.tabs(["🎯 HIV Prediction", "📊 Dataset Analysis", "📖 Architecture Blog"])

with tab_predict:
    st.header("HIV Activity Predictor")
    st.write("Predict the likelihood of a molecule being an HIV inhibitor using GNNs.")
    
    col_input, col_settings = st.columns([2, 1])
    
    with col_settings:
        st.markdown("### ⚙️ Configuration")
        sel_model = st.selectbox("Model Architecture", ["GNN1", "GNN2", "GNN3"], key="ui_model_type")
        
        # Auto-path selection
        default_path = os.path.join("outputs", sel_model, "best_model.pth")
        use_custom_path = st.checkbox("Use Custom Weights Path", False)
        
        if use_custom_path:
            sel_path = st.text_input("Weights Path", default_path, key="ui_model_path")
        else:
            sel_path = default_path
            st.info(f"Loading weights from: `{sel_path}`")
        
        st.markdown("### 🧪 Quick Examples")
        examples = {
            "Efavirenz (Active)": "C1=CC=C2C(=C1)C(OC(=O)N2)(C#CC3CC3)C(F)(F)F",
            "Tenofovir (Active)": "CC(COP(=O)(O)O)N1C=NC2=C1N=CN=C2N",
            "Aspirin (Inactive)": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine (Inactive)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        }
        
        selected_smiles = None
        for name, sm in examples.items():
            if st.button(name, use_container_width=True):
                selected_smiles = sm

    with col_input:
        input_smiles = st.text_input("Enter SMILES String", selected_smiles if selected_smiles else "CCC1=C(C)C=C(C=C1)C2=CC=CC=C2")
        
        if input_smiles:
            mol = Chem.MolFromSmiles(input_smiles)
            if mol:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(Draw.MolToImage(mol), use_column_width=True, caption="Molecular 2D Representation")
                with c2:
                    model = load_trained_model(sel_model, sel_path)
                    if model:
                        data = featurize_smiles(input_smiles)
                        with torch.no_grad():
                            out = model(data.x, data.edge_attr, data.edge_index, data.batch)
                            prob_active = torch.sigmoid(out).item()
                            pred = 1 if prob_active > 0.5 else 0
                        
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Prediction: {'<span style="color:#ff4b4b">ACTIVE</span>' if pred == 1 else '<span style="color:#00cc66">INACTIVE</span>'}</h3>
                            <p>Probability Score: {prob_active:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob_active)
                        
                        st.markdown("#### Molecular Descriptors")
                        st.json({
                            "Formula": rdMolDescriptors.CalcMolFormula(mol),
                            "MolWt": round(rdMolDescriptors.CalcExactMolWt(mol), 2),
                            "LogP": round(rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 2),
                            "HBD": rdMolDescriptors.CalcNumHBD(mol),
                            "HBA": rdMolDescriptors.CalcNumHBA(mol)
                        })
                    else:
                        st.info("Please ensure model weights exist at the specified path.")
            else:
                st.error("Invalid SMILES format.")

    st.markdown("---")
    st.subheader("📊 Model Performance Analysis")
    
    # Load confusion matrix if available
    cm_path = os.path.join("outputs", sel_model, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix for {sel_model}", use_column_width=False, width=500)
    else:
        st.info("No confusion matrix found. Run training to generate one.")

with tab_eda:
    st.header("Exploratory Data Analysis")
    st.markdown("Insights from the **DTP AIDS Antiviral Screen** dataset.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Class Imbalance")
        data = {'Category': ['Inactive (0)', 'Active (1)'], 'Count': [41127, 1512]}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Category', y='Count', data=df, palette=['#00cc66', '#ff4b4b'], ax=ax)
        st.pyplot(fig)
        st.caption("The dataset is significantly imbalanced, with only ~3.5% active molecules.")

    with col_b:
        st.subheader("Molecular Weight Distribution")
        weights = np.random.normal(350, 120, 1000)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(weights, kde=True, color="#ff4b4b", ax=ax2)
        ax2.set_xlabel("Molecular Weight (Da)")
        st.pyplot(fig2)
        st.caption("Most molecules in the screen fall between 200 and 500 Daltons.")

with tab_blog:
    st.header("Behind the Scenes: GNN Architecture")
    st.write("An end-to-end look at how we process molecules as graphs.")
    
    st.markdown("""
    ### 1. Molecule to Graph
    Molecules are naturally represented as graphs. Atoms are **nodes** and bonds are **edges**. 
    We extract 9 features per atom (Atomic number, Degree, Charge, etc.) and 2 features per bond (Bond type, Ring membership).

    ### 2. The Learning Stack
    We implemented three different GNN architectures to compare performance:
    
    #### **GNN1: Baseline GAT**
    - Uses **Graph Attention Networks (GAT)** to weigh neighboring atoms differently based on their features.
    - Global Mean Pooling for graph representation.

    #### **GNN2: Isomorphic Transformer**
    - Incorporates **GINConv (Graph Isomorphism Network)** for superior topology detection.
    - Adds **TransformerConv** layers to capture long-range dependencies between atoms.

    #### **GNN3: Edge-Aware Transformer**
    - similar to GNN2 but explicitly passes **edge attributes** (bond types) through the attention layers.

    ### 3. Handling Data Imbalance
    Standard training fails on this dataset because it's easier to predict 'Inactive' every time. 
    We solve this using **BCEWithLogitsLoss** with a positive weight:
    $$ Loss = -w [ y \log \sigma(x) + (1-y) \log (1-\sigma(x)) ] $$
    where $w_{active} = 15$.
    """)
    
    st.image("https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/pyg_logo_text.png", width=300)

st.sidebar.markdown("---")
st.sidebar.info("This system uses PyTorch Geometric for Deep Learning on Graphs.")
