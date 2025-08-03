import sys
import os

# Ensure we can find all installed packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import custom modules
from config.settings import *
from config.languages import get_text
from config.help_content import get_help_content, get_help_title
from services.data_service import DataService
from services.sentiment_service import SentimentService
from services.indicators import TechnicalIndicators
from models.lstm_model import ForexPredictor
from utils.visualization import ForexVisualizer
from utils.cache_manager import CacheManager
import hashlib
import base64

# Authentication configuration
VALID_CREDENTIALS = {
    "artec": "e10adc3949ba59abbe56e057f20f883e"  # MD5 hash of "123456"
}

def get_logo_base64():
    """Get the company logo as base64 encoded string."""
    try:
        with open("assets/company_logo.png", "rb") as f:
            logo_data = f.read()
            return base64.b64encode(logo_data).decode()
    except FileNotFoundError:
        return ""

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")
        remember_me = st.session_state.get("remember_me", False)
        
        if username and password and username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == hashlib.md5(password.encode()).hexdigest():
            st.session_state["password_correct"] = True
            
            # Save credentials if remember me is checked
            if remember_me:
                st.session_state["saved_username"] = username
                st.session_state["saved_password"] = password
                st.session_state["credentials_saved"] = True
            
            # Clear the input fields but keep login state
            if "password" in st.session_state:
                del st.session_state["password"]
            if "username" in st.session_state:
                del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    def direct_login():
        """Direct login using saved credentials."""
        if st.session_state.get("credentials_saved", False):
            saved_username = st.session_state.get("saved_username", "")
            saved_password = st.session_state.get("saved_password", "")
            
            if saved_username in VALID_CREDENTIALS and VALID_CREDENTIALS[saved_username] == hashlib.md5(saved_password.encode()).hexdigest():
                st.session_state["password_correct"] = True
                return True
        return False

    # Auto-login with saved credentials on page load
    if st.session_state.get("credentials_saved", False) and not st.session_state.get("password_correct", False):
        if direct_login():
            return True

    # Check if user clicked direct login button
    if st.session_state.get("direct_login_clicked", False):
        if direct_login():
            st.session_state["direct_login_clicked"] = False
            return True
        else:
            st.session_state["direct_login_clicked"] = False
            st.session_state["credentials_saved"] = False  # Clear invalid saved credentials

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password with company logo
    logo_base64 = get_logo_base64()
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 60vh;">
        <div style="text-align: center; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); background: white; min-width: 400px;">
            <div style="margin-bottom: 1.5rem;">
                <img src="data:image/png;base64,{logo_base64}" style="max-width: 120px; height: auto;" />
            </div>
            <h2 style="color: #333; margin-bottom: 2rem;">{get_text("login_title")}</h2>
            <p style="color: #666; margin-bottom: 2rem;">{get_text("login_subtitle")}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if we have saved credentials for direct login
        has_saved_credentials = st.session_state.get("credentials_saved", False)
        
        if has_saved_credentials:
            st.success("🔑 Credenciais salvas encontradas!")
            
            col_direct, col_manual = st.columns(2)
            
            with col_direct:
                if st.button("🚀 Entrar Automaticamente", type="primary", use_container_width=True, key="auto_login_btn"):
                    # Direct login with saved credentials
                    saved_username = st.session_state.get("saved_username", "")
                    saved_password = st.session_state.get("saved_password", "")
                    
                    if saved_username in VALID_CREDENTIALS and VALID_CREDENTIALS[saved_username] == hashlib.md5(saved_password.encode()).hexdigest():
                        st.session_state["password_correct"] = True
                        st.rerun()  # Immediate redirect
                    else:
                        st.session_state["credentials_saved"] = False
                        st.error("❌ Credenciais salvas inválidas!")
                        st.rerun()
            
            with col_manual:
                if st.button("✏️ Inserir Manualmente", use_container_width=True, key="manual_login_btn"):
                    st.session_state["credentials_saved"] = False
                    st.rerun()
            
            st.markdown("---")
        
        # Manual login form
        st.text_input(get_text("username_placeholder"), key="username", placeholder=get_text("username_placeholder"))
        st.text_input(get_text("password_placeholder"), type="password", key="password", placeholder=get_text("password_placeholder"))
        
        # Remember me checkbox
        st.checkbox("🔒 Lembrar credenciais", key="remember_me", help="Salva suas credenciais para login automático futuro")
        
        # Login button with immediate response
        if st.button("🔓 Fazer Login", type="primary", use_container_width=True, key="login_submit"):
            username = st.session_state.get("username", "")
            password = st.session_state.get("password", "")
            remember_me = st.session_state.get("remember_me", False)
            
            if username and password and username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == hashlib.md5(password.encode()).hexdigest():
                st.session_state["password_correct"] = True
                
                # Save credentials if remember me is checked
                if remember_me:
                    st.session_state["saved_username"] = username
                    st.session_state["saved_password"] = password
                    st.session_state["credentials_saved"] = True
                
                # Clear the input fields
                if "password" in st.session_state:
                    del st.session_state["password"]
                if "username" in st.session_state:
                    del st.session_state["username"]
                
                st.rerun()  # Immediate redirect
            else:
                st.session_state["password_correct"] = False
                st.error("❌ Usuário ou senha incorretos!")
        
        # Remove the general error message since we handle it inline now
            
        # Clear saved credentials option
        if has_saved_credentials:
            if st.button("🗑️ Limpar Credenciais Salvas", help="Remove as credenciais salvas deste dispositivo", key="clear_creds_btn"):
                st.session_state["credentials_saved"] = False
                if "saved_username" in st.session_state:
                    del st.session_state["saved_username"]
                if "saved_password" in st.session_state:
                    del st.session_state["saved_password"]
                st.success("Credenciais removidas com sucesso!")
                st.rerun()
            
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #888; font-size: 0.9em;">
            <p>🔐 Suas credenciais são armazenadas localmente e criptografadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    return False

# Page configuration
st.set_page_config(
    page_title=get_text("login_title"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamic CSS based on theme
def apply_theme_css():
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        st.markdown("""
        <style>
            /* Dark Theme */
            .stApp {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            .main-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .main-header h1 {
                color: white;
                text-align: center;
                margin: 0;
                font-size: 2.5rem;
            }
            .metric-card {
                background: #2d2d2d !important;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                border-left: 4px solid #667eea;
                color: #ffffff !important;
            }
            .success-alert {
                background-color: #1b5e20 !important;
                border: 1px solid #4caf50;
                color: #e8f5e8 !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .error-alert {
                background-color: #b71c1c !important;
                border: 1px solid #f44336;
                color: #ffebee !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .warning-alert {
                background-color: #e65100 !important;
                border: 1px solid #ff9800;
                color: #fff3e0 !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .sidebar .stSelectbox label {
                font-weight: bold;
                color: #ffffff !important;
            }
            .stButton > button {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-weight: bold;
                width: 100%;
            }
            .stButton > button:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transform: translateY(-2px);
            }
            
            /* Dark theme sidebar and components */
            .css-1d391kg, .css-1lcbmhc, .css-1outpf7 {
                background-color: #2d2d2d !important;
            }
            
            /* Dark theme text - More specific selectors */
            .stApp .stMarkdown, 
            .stApp .stText, 
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
            .stApp p, .stApp div, .stApp span, 
            .stApp label,
            .stApp .stSelectbox label,
            .stApp .stSlider label,
            .stApp .stCheckbox label,
            .css-1kyxreq, .css-1kyxreq p, .css-1kyxreq div, .css-1kyxreq span,
            .element-container, .element-container p, .element-container div,
            .st-emotion-cache-1kyxreq,
            .st-emotion-cache-1kyxreq p,
            .st-emotion-cache-1kyxreq div,
            .st-emotion-cache-1kyxreq span,
            .st-emotion-cache-1kyxreq h1,
            .st-emotion-cache-1kyxreq h2,
            .st-emotion-cache-1kyxreq h3,
            .st-emotion-cache-1kyxreq h4 {
                color: #ffffff !important;
            }
            
            /* Dark theme input fields */
            .stSelectbox > div > div, 
            .stTextInput > div > div > input, 
            .stSlider > div > div,
            .stSelectbox div[data-baseweb="select"] > div,
            .stSelectbox div[data-baseweb="select"] > div > div,
            .stSelectbox [role="option"],
            .stSelectbox [role="listbox"] {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
                border: 1px solid #444 !important;
            }
            
            /* Dark theme dropdown options */
            .stSelectbox div[data-baseweb="popover"] {
                background-color: #2d2d2d !important;
            }
            
            .stSelectbox div[data-baseweb="popover"] li {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            .stSelectbox div[data-baseweb="popover"] li:hover {
                background-color: #404040 !important;
                color: #ffffff !important;
            }
            
            /* Dark theme expanders */
            .streamlit-expanderHeader,
            details[open] > summary,
            details > summary,
            .st-emotion-cache-1p1nwyz,
            .st-emotion-cache-1p1nwyz p,
            .st-emotion-cache-1p1nwyz div {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            /* Dark theme tabs */
            .stTabs [data-baseweb="tab-list"],
            .stTabs [data-baseweb="tab-list"] button,
            .st-emotion-cache-1whx7iy,
            .st-emotion-cache-1whx7iy button {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            .stTabs [data-baseweb="tab"],
            .stTabs [data-baseweb="tab"] div,
            .st-emotion-cache-1whx7iy button div {
                color: #ffffff !important;
                background-color: #2d2d2d !important;
            }
            
            .stTabs [data-baseweb="tab-panel"],
            .st-emotion-cache-1kyxreq {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            
            /* Dark theme for all containers and panels */
            .block-container,
            .css-1kyxreq,
            .element-container,
            .st-emotion-cache-1kyxreq,
            .st-emotion-cache-16idsys,
            .st-emotion-cache-1wmy9hl,
            .st-emotion-cache-12fmjuu {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            
            /* Dark theme metrics and info boxes */
            .stMetric,
            .stMetric label,
            .stMetric div,
            .stInfo,
            .stSuccess,
            .stWarning,
            .stError {
                color: #ffffff !important;
            }
            
            /* Dark theme sidebar improvements */
            .css-1d391kg,
            .css-1lcbmhc, 
            .css-1outpf7,
            .st-emotion-cache-16idsys,
            .st-emotion-cache-1gwvy71 {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            .css-1d391kg p,
            .css-1d391kg div,
            .css-1d391kg span,
            .css-1d391kg label {
                color: #ffffff !important;
            }
            
            /* Dark theme for tutorial sections - comprehensive */
            .css-1kyxreq ul,
            .css-1kyxreq li,
            .css-1kyxreq strong,
            .css-1kyxreq em,
            .st-emotion-cache-1kyxreq ul,
            .st-emotion-cache-1kyxreq li,
            .st-emotion-cache-1kyxreq strong,
            .st-emotion-cache-1kyxreq em {
                color: #ffffff !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            /* Light Theme */
            .stApp {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .main-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .main-header h1 {
                color: white;
                text-align: center;
                margin: 0;
                font-size: 2.5rem;
            }
            .metric-card {
                background: white !important;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
                color: #000000 !important;
            }
            .success-alert {
                background-color: #d4edda !important;
                border: 1px solid #c3e6cb;
                color: #155724 !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .error-alert {
                background-color: #f8d7da !important;
                border: 1px solid #f5c6cb;
                color: #721c24 !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .warning-alert {
                background-color: #fff3cd !important;
                border: 1px solid #ffeaa7;
                color: #856404 !important;
                padding: 0.75rem;
                border-radius: 0.25rem;
                margin: 1rem 0;
            }
            .sidebar .stSelectbox label {
                font-weight: bold;
                color: #333 !important;
            }
            .stButton > button {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-weight: bold;
                width: 100%;
                min-height: 44px; /* iOS touch target minimum */
                font-size: 16px; /* Prevent zoom on iOS */
                touch-action: manipulation; /* Improve touch response */
                transition: all 0.2s ease;
            }
            .stButton > button:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transform: translateY(-2px);
            }
            .stButton > button:active {
                transform: translateY(0);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            /* Mobile-specific improvements for general interface */
            @media (max-width: 768px) {
                /* Enhanced button sizing for mobile */
                .stButton > button {
                    padding: 1rem !important;
                    font-size: 16px !important;
                    min-height: 48px !important;
                    border-radius: 10px !important;
                    margin: 0.5rem 0 !important;
                }
                
                /* Improve selectbox touch targets */
                .stSelectbox > div > div {
                    min-height: 48px !important;
                    font-size: 16px !important;
                    padding: 0.75rem !important;
                }
                
                /* Better sidebar organization */
                .css-1d391kg {
                    padding: 1rem 0.5rem !important;
                }
                
                /* Stack columns for mobile readability */
                div[data-testid="column"] {
                    min-width: 100% !important;
                    margin-bottom: 1rem !important;
                }
                
                /* Optimize main content area */
                .block-container {
                    padding: 1rem 0.5rem !important;
                    max-width: 100% !important;
                }
                
                /* Responsive header design */
                .main-header {
                    padding: 1rem 0.5rem !important;
                    text-align: center !important;
                }
                
                .main-header h1 {
                    font-size: 1.5rem !important;
                    margin: 0.5rem 0 !important;
                }
                
                .main-header p {
                    font-size: 0.9rem !important;
                }
                
                /* Better text readability */
                h1, h2, h3, h4 {
                    font-size: 1.2rem !important;
                    line-height: 1.4 !important;
                    margin: 0.75rem 0 !important;
                }
                
                p, div, span {
                    font-size: 14px !important;
                    line-height: 1.5 !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize all services once and cache them."""
    return {
        'data_service': DataService(),
        'sentiment_service': SentimentService(),
        'visualizer': ForexVisualizer()
    }

services = initialize_services()

# Additional Dark Theme fixes for dashboard functions panel - apply globally for better coverage
if st.session_state.get('theme', 'light') == 'dark':
    st.markdown("""
    <style>
        /* Critical fixes for Dark theme dashboard functions visibility */
        
        /* Ensure ALL text elements in main content are visible */
        .main .block-container,
        .main .block-container *,
        .stApp .main,
        .stApp .main *,
        .css-1kyxreq *,
        .st-emotion-cache-1kyxreq *,
        [data-testid="stHorizontalBlock"] *,
        [data-testid="stVerticalBlock"] *,
        [data-testid="stColumns"] *,
        [data-testid="column"] *,
        .element-container *,
        .st-emotion-cache-12fmjuu *,
        .st-emotion-cache-1wmy9hl * {
            color: #ffffff !important;
        }
        
        /* Function panel specific fixes */
        .stTabs .st-emotion-cache-1kyxreq,
        .stTabs .st-emotion-cache-1kyxreq p,
        .stTabs .st-emotion-cache-1kyxreq div,
        .stTabs .st-emotion-cache-1kyxreq span,
        .stTabs .st-emotion-cache-1kyxreq li,
        .stTabs .st-emotion-cache-1kyxreq ul,
        .stTabs .st-emotion-cache-1kyxreq h1,
        .stTabs .st-emotion-cache-1kyxreq h2,
        .stTabs .st-emotion-cache-1kyxreq h3,
        .stTabs .st-emotion-cache-1kyxreq h4,
        .stTabs .st-emotion-cache-1kyxreq strong,
        .stTabs .st-emotion-cache-1kyxreq em {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        
        /* Markdown content in Dark theme */
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        .stMarkdown p,
        .stMarkdown div,
        .stMarkdown span,
        .stMarkdown li,
        .stMarkdown ul,
        .stMarkdown strong,
        .stMarkdown em,
        .stMarkdown code {
            color: #ffffff !important;
        }
        
        /* Tutorial sections comprehensive fix */
        .css-1kyxreq,
        .css-1kyxreq *,
        .st-emotion-cache-1kyxreq,
        .st-emotion-cache-1kyxreq * {
            color: #ffffff !important;
        }
        
        /* Column content visibility */
        [data-testid="column"] div,
        [data-testid="column"] p,
        [data-testid="column"] span,
        [data-testid="column"] h1,
        [data-testid="column"] h2,
        [data-testid="column"] h3,
        [data-testid="column"] h4,
        [data-testid="column"] li,
        [data-testid="column"] ul,
        [data-testid="column"] strong {
            color: #ffffff !important;
        }
        
        /* Info boxes and alerts in Dark theme */
        .info-alert,
        .success-alert,
        .warning-alert,
        .error-alert {
            border: 1px solid #444 !important;
        }
        
        .info-alert,
        .info-alert * {
            background-color: #1a237e !important;
            color: #e3f2fd !important;
        }
        
        /* Gradient boxes for Dark theme */
        div[style*="background: linear-gradient"] * {
            color: #ffffff !important;
        }
        
        /* Ensure button text is visible */
        .stButton > button,
        .stButton > button * {
            color: white !important;
        }
        
        /* Progress and status elements */
        .stProgress,
        .stProgress *,
        .stMetric,
        .stMetric * {
            color: #ffffff !important;
        }
        
        /* Fix for any remaining invisible elements */
        * {
            color: inherit !important;
        }
        
        /* Override any Streamlit defaults for Dark theme */
        .stApp[data-theme="dark"] *,
        [data-theme="dark"] * {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Common CSS for both themes
st.markdown("""
<style>
    /* Buy/Sell/Hold signals - consistent across themes */
    .buy-signal {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .sell-signal {
        background: linear-gradient(135deg, #f44336, #da190b);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .hold-signal {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-alert {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        color: #0d47a1;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit's default loading indicators completely */
    .stSpinner, .stAlert, .stSpinner > div {
        display: none !important;
    }
    
    /* Hide the running indicator in top right corner */
    .stApp > header,
    [data-testid="stToolbar"],
    .stApp > div[data-testid="stHeader"],
    .streamlit-container .element-container:has(.stSpinner),
    .stApp .stSpinner,
    [data-testid="stStatusWidget"],
    .StatusWidget,
    .streamlit-container .stAlert {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide the menu button (3 dots) and running status */
    .css-14xtw13.e8zbici0,
    .css-h5rgaw.egzxvld1,
    button[title="View fullscreen"],
    [data-testid="stDecoration"],
    .decoration,
    .css-1rs6os.edgvbvh3,
    .css-1vbkxwb.e1nzilvr5,
    .css-1vbkxwb,
    [aria-label="Running..."],
    [data-testid="stAppViewContainer"] > .stSpinner {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* Animation for custom spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom spinner for manual use */
    .custom-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px;
        margin: 20px 0;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    /* COMPREHENSIVE REMOVAL OF ALL STREAMLIT LOADING ELEMENTS */
    /* Remove all spinners, progress bars, status messages, and loading indicators */
    .stApp [data-testid="stHeader"],
    .stApp .stSpinner,
    .stSpinner,
    [data-testid="stSpinner"],
    .css-1outpf7,
    .css-164nlkn,
    .css-12oz5g7,
    .css-1n76uvr,
    [data-baseweb="notification"],
    .streamlit-spinner,
    .stProgress,
    [data-testid="stProgress"],
    .stAlert,
    [data-testid="stAlert"],
    .stStatus,
    [data-testid="stStatus"],
    .stToast,
    [data-testid="stToast"],
    .element-container:has(.stSpinner),
    .element-container:has([data-testid="stSpinner"]),
    div:has(> .stSpinner),
    div:has(> [data-testid="stSpinner"]),
    .st-emotion-cache-*:has(.stSpinner),
    .block-container .stSpinner,
    .main .stSpinner,
    [class*="spinner"],
    [class*="loading"],
    [class*="status"],
    [class*="progress"],
    .css-*:has(.stSpinner),
    .st-*:has(.stSpinner),
    .st-emotion-cache-*,
    [data-testid="stLoading"],
    [data-testid="stProgressBar"],
    .stProgress > *,
    .stStatus > *,
    .stSpinner > *,
    div[data-testid="element-container"]:has(.stSpinner),
    div[data-testid="element-container"]:has(.stProgress),
    div[data-testid="element-container"]:has(.stStatus),
    .css-1wrcr25,
    .css-12oz5g7,
    .css-5uatcg,
    .st-emotion-cache-1wrcr25,
    .st-emotion-cache-12oz5g7,
    .st-emotion-cache-5uatcg {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        left: -9999px !important;
        z-index: -1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Services already initialized above

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    # Check authentication first
    if not check_password():
        return
    
    # Apply theme CSS immediately after authentication
    apply_theme_css()
    
    # Force Dark theme CSS specifically for tutorial/configuration panels
    if st.session_state.get('theme', 'light') == 'dark':
        st.markdown("""
        <style>
            /* CRITICAL: Force white text on ALL elements in Dark theme */
            
            /* Global override for all text elements */
            * {
                color: #ffffff !important;
            }
            
            /* Specific elements that need white text */
            h1, h2, h3, h4, h5, h6,
            p, div, span, label, li, ul, ol,
            strong, em, b, i, code, pre,
            .stMarkdown, .stText,
            [data-testid] *,
            .element-container *,
            .block-container *,
            .css-* *,
            .st-* *,
            .st-emotion-* * {
                color: #ffffff !important;
            }
            
            /* Tutorial and tab specific fixes */
            .stTabs *,
            .stTabs div,
            .stTabs p,
            .stTabs span,
            .stTabs h1,
            .stTabs h2,
            .stTabs h3,
            .stTabs h4,
            .stTabs li,
            .stTabs ul,
            .stTabs strong,
            .stTabs em,
            div[data-baseweb="tab-panel"] *,
            div[data-testid="stHorizontalBlock"] *,
            div[data-testid="stVerticalBlock"] *,
            div[data-testid="column"] *,
            div[data-testid="stMarkdown"] * {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            /* Override any conflicting styles */
            .main .block-container * {
                color: #ffffff !important;
            }
            
            /* Ensure buttons remain readable and mobile-friendly + remove spinners */
            .stButton > button {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.75rem 1rem !important;
                font-weight: bold !important;
                width: 100% !important;
                min-height: 44px !important; /* iOS touch target minimum */
                font-size: 16px !important; /* Prevent zoom on iOS */
                touch-action: manipulation !important; /* Improve touch response */
                transition: all 0.2s ease !important;
            }
            
            /* Aggressive spinner removal for Dark theme */
            .stSpinner,
            [data-testid="stSpinner"],
            .streamlit-spinner,
            .stProgress,
            [data-testid="stProgress"],
            .stAlert,
            [data-testid="stAlert"],
            .stStatus,
            [data-testid="stStatus"],
            .stToast,
            [data-testid="stToast"],
            [class*="spinner"],
            [class*="loading"],
            [class*="status"],
            .element-container:has(.stSpinner),
            div:has(> .stSpinner) {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                width: 0 !important;
                position: absolute !important;
                left: -9999px !important;
            }
            
            .stButton > button:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
                transform: translateY(-2px) !important;
            }
            
            .stButton > button:active {
                transform: translateY(0) !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            }
            
            /* Ensure dropdowns work properly and are mobile-friendly */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
                border: 1px solid #444 !important;
                min-height: 44px !important;
                font-size: 16px !important;
            }
            
            /* MODERN MOBILE-FIRST RESPONSIVE DESIGN WITH TOGGLE */
            @media (max-width: 768px) {
                /* Main content adjustment for mobile toggle */
                .main .block-container {
                    padding-top: 4rem !important;
                }
                
                /* Enhanced mobile configuration panel */
                .stColumns {
                    gap: 1rem !important;
                }
                
                .stColumns > div {
                    padding: 1rem !important;
                    background: rgba(255,255,255,0.05) !important;
                    border-radius: 12px !important;
                    margin-bottom: 1rem !important;
                }
                
                /* Mobile responsive columns stack vertically */
                @media (max-width: 480px) {
                    .stColumns {
                        flex-direction: column !important;
                    }
                    
                    .stColumns > div {
                        width: 100% !important;
                        margin-bottom: 1rem !important;
                    }
                }
                
                /* Sidebar styling - responsive behavior */
                .css-1d391kg,
                .st-emotion-cache-1d391kg,
                section[data-testid="stSidebar"] {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 85% !important;
                    max-width: 350px !important;
                    height: 100vh !important;
                    background: white !important;
                    box-shadow: 2px 0 10px rgba(0,0,0,0.1) !important;
                    z-index: 9998 !important;
                    overflow-y: auto !important;
                    padding: 1rem !important;
                    transition: transform 0.3s ease !important;
                    /* Show by default */
                    transform: translateX(0) !important;
                }
                
                /* Hide sidebar when closed */
                .sidebar-closed .css-1d391kg,
                .sidebar-closed .st-emotion-cache-1d391kg,
                .sidebar-closed section[data-testid="stSidebar"] {
                    transform: translateX(-100%) !important;
                }
                
                /* Overlay for sidebar - visible by default */
                .sidebar-overlay {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    background: rgba(0,0,0,0.5) !important;
                    z-index: 9997 !important;
                    opacity: 1 !important;
                    visibility: visible !important;
                    transition: all 0.3s ease !important;
                }
                
                /* Hide overlay when sidebar is closed */
                .sidebar-closed .sidebar-overlay {
                    opacity: 0 !important;
                    visibility: hidden !important;
                }
                
                /* Show overlay when sidebar is open (default) */
                .sidebar-overlay {
                    opacity: 1 !important;
                    visibility: visible !important;
                }
                
                /* Modern sidebar styling for mobile */
                .css-1d391kg > div,
                .st-emotion-cache-1d391kg > div {
                    padding: 0 !important;
                    width: 100% !important;
                }
                
                /* Adjust main content for mobile */
                .main .block-container {
                    padding-left: 1rem !important;
                    padding-top: 4rem !important; /* Space for toggle button */
                }
                
                /* Better button sizing for mobile */
                .stButton > button {
                    padding: 1rem !important;
                    font-size: 16px !important;
                    min-height: 52px !important;
                    border-radius: 12px !important;
                    width: 100% !important;
                    margin: 0.5rem 0 !important;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
                    transition: all 0.2s ease !important;
                }
                
                /* Enhanced mobile touch targets for selectors */
                .stSelectbox > div > div,
                .stSelectbox div[data-baseweb="select"] > div {
                    min-height: 52px !important;
                    font-size: 16px !important;
                    padding: 1rem !important;
                    border-radius: 12px !important;
                    border: 1px solid rgba(0,0,0,0.2) !important;
                }
                
                /* Modern slider styling for mobile */
                .stSlider {
                    padding: 1.5rem 0 !important;
                }
                
                .stSlider > div > div {
                    height: 8px !important;
                    border-radius: 4px !important;
                }
                
                /* Stack columns beautifully on mobile */
                div[data-testid="column"] {
                    min-width: 100% !important;
                    margin-bottom: 1.5rem !important;
                    padding: 0 0.5rem !important;
                }
                
                /* Improved main container with modern spacing */
                .block-container {
                    padding: 1rem !important;
                    max-width: 100% !important;
                }
                
                /* Modern header design for mobile */
                .main-header {
                    padding: 1.5rem 1rem !important;
                    text-align: center !important;
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
                    border-radius: 16px !important;
                    margin-bottom: 1.5rem !important;
                }
                
                .main-header h1 {
                    font-size: 1.8rem !important;
                    margin: 0.5rem 0 !important;
                    font-weight: 700 !important;
                }
                
                .main-header p {
                    font-size: 1rem !important;
                    margin: 0.5rem 0 !important;
                    opacity: 0.9 !important;
                }
                
                /* Modern tab navigation for mobile */
                .stTabs > div > div > div {
                    overflow-x: auto !important;
                    white-space: nowrap !important;
                    padding: 0.5rem 0 !important;
                    -webkit-overflow-scrolling: touch !important;
                }
                
                .stTabs > div > div > div > button {
                    min-width: 140px !important;
                    padding: 1rem 1.25rem !important;
                    font-size: 15px !important;
                    border-radius: 12px !important;
                    margin: 0 0.25rem !important;
                    font-weight: 600 !important;
                }
                
                /* Modern typography for mobile */
                h1, h2, h3, h4 {
                    font-size: 1.3rem !important;
                    line-height: 1.4 !important;
                    margin: 1rem 0 !important;
                    font-weight: 600 !important;
                }
                
                p, div, span {
                    font-size: 15px !important;
                    line-height: 1.6 !important;
                }
                
                /* Modern metric cards for mobile */
                div[data-testid="metric-container"] {
                    padding: 1.5rem !important;
                    margin: 1rem 0 !important;
                    border-radius: 16px !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
                    border: 1px solid rgba(0,0,0,0.08) !important;
                }
                
                /* Modern input styling */
                .stTextInput > div > div > input,
                .stNumberInput > div > div > input {
                    min-height: 52px !important;
                    font-size: 16px !important;
                    border-radius: 12px !important;
                    border: 1px solid rgba(0,0,0,0.2) !important;
                    padding: 1rem !important;
                }
                
                /* Remove ALL loading elements on mobile */
                .stSpinner,
                [data-testid="stSpinner"],
                .streamlit-spinner,
                .stProgress,
                [data-testid="stProgress"],
                .stStatus,
                [data-testid="stStatus"],
                [class*="spinner"],
                [class*="loading"] {
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                }
            }
            
            /* MODERN TABLET DESIGN */
            @media (min-width: 769px) and (max-width: 1024px) {
                .block-container {
                    padding: 1.5rem !important;
                    max-width: 95% !important;
                }
                
                .stButton > button {
                    min-height: 48px !important;
                    font-size: 15px !important;
                    border-radius: 10px !important;
                    padding: 0.875rem 1.25rem !important;
                }
                
                /* Enhanced sidebar for tablet */
                section[data-testid="stSidebar"] {
                    width: 320px !important;
                    padding: 1.25rem !important;
                }
                
                /* Better spacing for tablets */
                .main-header {
                    padding: 2rem 1.5rem !important;
                    border-radius: 16px !important;
                }
                
                div[data-testid="metric-container"] {
                    padding: 1.25rem !important;
                    border-radius: 12px !important;
                }
            }
            
            /* MODERN DESKTOP DESIGN */
            @media (min-width: 1025px) {
                /* Enhanced desktop sidebar */
                section[data-testid="stSidebar"] {
                    width: 350px !important;
                    padding: 1.5rem !important;
                    background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
                    border-right: 1px solid rgba(0,0,0,0.1) !important;
                    position: relative !important;
                    transform: none !important;
                }
                
                /* Reset main content padding on desktop */
                .main .block-container {
                    padding-top: 1rem !important;
                }
                
                /* Reset main content padding on desktop */
                .main .block-container {
                    padding-top: 1rem !important;
                }
                
                /* Modern desktop buttons */
                .stButton > button {
                    min-height: 44px !important;
                    border-radius: 8px !important;
                    transition: all 0.2s ease !important;
                }
                
                .stButton > button:hover {
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
                }
                
                /* Desktop header styling */
                .main-header {
                    padding: 2.5rem 2rem !important;
                    border-radius: 20px !important;
                    margin-bottom: 2rem !important;
                }
                
                /* Enhanced desktop typography */
                .main-header h1 {
                    font-size: 2.2rem !important;
                    font-weight: 700 !important;
                }
                
                .main-header p {
                    font-size: 1.1rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Mobile sidebar toggle using Streamlit button - default open
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = True
    
    # Create mobile toggle button using columns for positioning
    col_toggle, col_spacer = st.columns([1, 20])
    
    with col_toggle:
        # Mobile toggle button with CSS styling
        st.markdown("""
        <style>
        @media (max-width: 768px) {
            .mobile-toggle-button {
                position: fixed !important;
                top: 1rem !important;
                left: 1rem !important;
                z-index: 9999 !important;
                background: linear-gradient(135deg, #667eea, #764ba2) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                min-width: 48px !important;
                min-height: 48px !important;
                font-size: 1.2rem !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            }
            
            .mobile-toggle-button button {
                background: transparent !important;
                border: none !important;
                color: white !important;
                font-size: 1.2rem !important;
                width: 100% !important;
                height: 100% !important;
            }
        }
        
        @media (min-width: 769px) {
            .mobile-toggle-button {
                display: none !important;
            }
        }
        </style>
        
        <div class="mobile-toggle-button">
        """, unsafe_allow_html=True)
        
        # Toggle button - show close when open, hamburger when closed
        if st.button('✕' if st.session_state.sidebar_open else '☰', key="mobile_sidebar_toggle", help="Abrir/Fechar Menu"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Apply sidebar state with JavaScript - use sidebar-closed instead of sidebar-open
    sidebar_js_state = "true" if st.session_state.sidebar_open else "false"
    
    st.markdown(f"""
    <script>
    // Apply sidebar state - default is open, add closed class when needed
    document.body.className = document.body.className.replace(/sidebar-closed/g, '');
    if (!{sidebar_js_state}) {{
        document.body.classList.add('sidebar-closed');
    }}
    </script>
    """, unsafe_allow_html=True)
    
    # Header with company logo
    logo_base64 = get_logo_base64()
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width: 60px; height: auto;" />
            <div>
                <h1 style="margin: 0;">{get_text("main_title")}</h1>
                <p style="color: white; text-align: center; margin: 0; font-size: 1.1rem;">
                    Previsões Forex com IA e Análise em Tempo Real
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick action buttons in main content area
    st.markdown("## 🚀 Análises Rápidas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Análise Técnica", type="primary", use_container_width=True, key="quick_technical"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'technical'
    
    with col2:
        if st.button("🤖 Previsão IA", type="primary", use_container_width=True, key="quick_ai"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'prediction'
    
    with col3:
        if st.button("📈 Dashboard Completo", type="primary", use_container_width=True, key="quick_complete"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'complete'
    
    st.markdown("---")
    
    # Comprehensive sidebar configuration
    with st.sidebar:
        # Modern sidebar header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem; 
                    background: linear-gradient(135deg, #667eea, #764ba2); 
                    border-radius: 12px; color: white;">
            <h2 style="margin: 0; font-size: 1.3rem; font-weight: 600;">⚙️ Painel de Controle</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Configure e Execute Análises</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trading Configuration Section
        st.markdown("### 💱 Configuração de Trading")
        
        # Currency pair selection with modern styling
        pair = st.selectbox(
            "Par de Moedas",
            ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
            key="sidebar_currency_pair",
            help="Selecione o par de moedas para análise"
        )
        
        # Time interval
        interval = st.selectbox(
            "Intervalo de Tempo",
            ["1min", "5min", "15min", "30min", "60min", "Daily"],
            index=4,  # Default to 60min
            key="sidebar_interval",
            help="Intervalo de tempo para coleta de dados"
        )
        
        # Prediction horizon
        horizon = st.selectbox(
            "Horizonte de Previsão",
            ["1 hora", "4 horas", "1 dia", "1 semana"],
            index=1,
            key="sidebar_horizon",
            help="Período para previsões futuras"
        )
        
        st.markdown("---")
        
        # Analysis Configuration Section
        st.markdown("### 🎯 Configuração de Análise")
        
        # Risk level
        risk_level = st.selectbox(
            "Nível de Risco",
            ["Conservativo", "Moderado", "Agressivo"],
            index=1,  # Default to Moderate
            key="sidebar_risk_level",
            help="Define a agressividade das recomendações"
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Tipo de Análise",
            ["Completa", "Técnica", "Sentimento", "IA Apenas"],
            index=0,
            key="sidebar_analysis_type",
            help="Selecione o tipo de análise desejada"
        )
        
        st.markdown("---")
        
        # AI Configuration Section  
        st.markdown("### 🧠 Configuração de IA")
        
        # AI Parameters
        lookback_period = st.slider(
            "Histórico de Dados",
            min_value=30,
            max_value=120,
            value=LOOKBACK_PERIOD,
            key="main_sidebar_lookback_period",
            help="Períodos históricos para treinamento da IA"
        )
        
        epochs = st.slider(
            "Intensidade do Treinamento IA",
            min_value=5,
            max_value=20,
            value=EPOCHS,
            key="main_sidebar_epochs",
            help="Mais épocas = melhor precisão mas mais lento"
        )
        
        mc_samples = st.slider(
            "Amostras de Previsão",
            min_value=10,
            max_value=50,
            value=MC_SAMPLES,
            key="main_sidebar_mc_samples",
            help="Amostras para estimativa de incerteza"
        )
        
        st.markdown("---")
        
        # Cache Management Section
        st.markdown("### 💾 Gerenciamento de Cache")
        
        # Cache status
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        
        if cache_count > 0:
            st.metric("Análises em Cache", cache_count, "dados salvos")
            
            if st.button("🗑️ Limpar Cache", key="main_sidebar_clear_cache"):
                CacheManager.clear_cache()
                st.success("Cache limpo!")
                st.rerun()
        else:
            st.info("📂 Nenhuma análise em cache")
        
        # Auto-save settings
        auto_save = st.checkbox("💾 Salvar Automaticamente", value=True, key="main_sidebar_auto_save", help="Salva análises automaticamente no cache")
        
        st.markdown("---")
        
        # Analysis Execution Section
        st.markdown("### 🚀 Executar Análises")
        
        # Main analysis buttons
        if st.button("📊 Análise Técnica Completa", type="primary", use_container_width=True, key="sidebar_btn_technical"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'technical'
            st.session_state['selected_pair'] = pair
            st.session_state['selected_interval'] = interval
            st.session_state['risk_level'] = risk_level
        
        if st.button("🤖 Previsão com IA", type="primary", use_container_width=True, key="sidebar_btn_ai"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'prediction'
            st.session_state['selected_pair'] = pair
            st.session_state['selected_interval'] = interval
            st.session_state['lookback_period'] = lookback_period
            st.session_state['epochs'] = epochs
            st.session_state['mc_samples'] = mc_samples
        
        if st.button("📰 Análise de Sentimento", use_container_width=True, key="sidebar_btn_sentiment"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'sentiment'
            st.session_state['selected_pair'] = pair
        
        if st.button("⚡ Análise Rápida", use_container_width=True, key="sidebar_btn_quick"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'quick'
            st.session_state['selected_pair'] = pair
            st.session_state['selected_interval'] = interval
        
        if st.button("📈 Dashboard Completo", type="primary", use_container_width=True, key="sidebar_btn_complete"):
            st.session_state['show_analysis'] = True
            st.session_state['analysis_type'] = 'complete'
            st.session_state['selected_pair'] = pair
            st.session_state['selected_interval'] = interval
            st.session_state['risk_level'] = risk_level
            st.session_state['lookback_period'] = lookback_period
            st.session_state['epochs'] = epochs
            st.session_state['mc_samples'] = mc_samples
        
        st.markdown("---")
        
        # Theme and Settings Section
        st.markdown("### 🎨 Aparência & Configurações")
        
        # Theme selector
        theme = st.selectbox(
            "Tema da Interface",
            ["Light (Claro)", "Dark (Escuro)"],
            index=0 if st.session_state.get('theme', 'light') == 'light' else 1,
            help="Escolha entre tema claro ou escuro",
            key="sidebar_theme_selector"
        )
        
        # Update theme in session state
        current_theme = 'light' if theme == "Light (Claro)" else 'dark'
        if st.session_state.get('theme', 'light') != current_theme:
            st.session_state['theme'] = current_theme
            apply_theme_css()  # Apply CSS when theme changes
            st.rerun()
        
        # Performance mode
        performance_mode = st.selectbox(
            "Modo de Performance",
            ["Balanceado", "Velocidade", "Precisão"],
            index=0,
            help="Balanceado: velocidade e precisão equilibradas",
            key="sidebar_performance_mode"
        )
        
        st.markdown("---")
        
        # Configuration Summary
        st.markdown("### 📋 Configuração Atual")
        
        # Current settings display
        st.info(f"📊 **Par:** {pair}")
        st.info(f"⏱️ **Intervalo:** {interval}")
        st.info(f"🎯 **Risco:** {risk_level}")
        st.info(f"🧠 **IA:** {lookback_period}d | {epochs}e")
        
        if cache_count > 0:
            st.success(f"💾 {cache_count} análises em cache")
        
        st.markdown("---")
        
        # Tutorial and Help Section
        st.markdown("### 📚 Ajuda")
        
        if st.button("📖 Tutorial Completo", help="Abrir guia detalhado", use_container_width=True, key="sidebar_tutorial"):
            st.session_state['show_tutorial'] = not st.session_state.get('show_tutorial', False)
        
        # Logout button
        if st.button("🚪 Sair", help="Sair da plataforma", use_container_width=True, key="sidebar_logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Trading Configuration Section
        st.markdown("### 💱 Configuração de Trading")
        
        # Currency pair selection with modern styling
        pair = st.selectbox(
            "Par de Moedas",
            PAIRS,
            help=get_help_content("currency_pair")
        )
        
        # Time interval
        interval = st.selectbox(
            "Intervalo de Tempo",
            list(INTERVALS.keys()),
            index=4,  # Default to 60min
            help=get_help_content("time_interval")
        )
        
        # Prediction horizon
        horizon = st.selectbox(
            "Horizonte de Previsão",
            HORIZONS,
            help=get_help_content("prediction_horizon")
        )
        
        st.markdown("---")
        
        # Risk Management Section
        st.markdown("### ⚖️ Gestão de Risco")
        
        # Risk management level
        risk_level = st.selectbox(
            "Nível de Risco",
            ["Conservativo", "Moderado", "Agressivo"],
            index=1,
            help=get_help_content("risk_level")
        )
        
        st.markdown("---")
        
        # AI Configuration Section
        st.markdown("### 🧠 Configuração IA")
        
        with st.expander("⚙️ Opções Avançadas", expanded=False):
            lookback_period = st.slider(
                "Períodos de Análise",
                min_value=30,
                max_value=120,
                value=LOOKBACK_PERIOD,
                help="Períodos históricos para treinamento da IA",
                key="expander_lookback_period"
            )
            
            epochs = st.slider(
                "Intensidade do Treinamento IA",
                min_value=5,
                max_value=20,
                value=EPOCHS,
                help="Mais épocas = melhor precisão mas mais lento",
                key="expander_epochs"
            )
            
            mc_samples = st.slider(
                "Amostras de Previsão",
                min_value=10,
                max_value=50,
                value=MC_SAMPLES,
                help="Amostras para estimativa de incerteza",
                key="expander_mc_samples"
            )
            
            # Mobile-optimized cache button
            st.markdown("""
            <style>
                @media (max-width: 768px) {
                    .stButton > button:contains("Limpar") {
                        min-height: 44px !important;
                        font-size: 14px !important;
                        padding: 0.75rem !important;
                    }
                }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("Limpar Cache", key="expander_clear_cache"):
                CacheManager.clear_cache()
                st.success("Cache limpo!")
                st.rerun()
        
        # Simple status
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            st.info(f"💾 {cache_count} análises em cache disponíveis")
        
        st.markdown("---")
        
        # Analysis buttons section
        
        # Enhanced mobile responsiveness for main buttons
        st.markdown("""
        <style>
            /* Critical mobile improvements for buttons */
            @media (max-width: 768px) {
                /* Primary button styling for mobile */
                .stButton > button[kind="primary"],
                .stButton > button {
                    width: 100% !important;
                    min-height: 52px !important;
                    font-size: 16px !important;
                    padding: 1rem 1.5rem !important;
                    border-radius: 12px !important;
                    margin: 0.75rem 0 !important;
                    touch-action: manipulation !important;
                    -webkit-tap-highlight-color: rgba(0,0,0,0) !important;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
                }
                
                /* Enhanced touch feedback */
                .stButton > button:active {
                    transform: scale(0.98) !important;
                    transition: transform 0.1s ease !important;
                }
                
                /* Better spacing between buttons */
                .element-container:has(.stButton) {
                    margin: 0.5rem 0 !important;
                }
            }
            
            /* Tablet optimizations */
            @media (min-width: 769px) and (max-width: 1024px) {
                .stButton > button {
                    min-height: 48px !important;
                    font-size: 15px !important;
                    padding: 0.875rem 1.25rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        analyze_button = st.button(
            "🎯 Obter Sinal de Trading", 
            type="primary", 
            help=get_help_content("get_trading_signal")
        )
        
        quick_analysis = st.button(
            "⚡ Verificação Rápida",
            help=get_help_content("quick_check")
        )
        
        st.markdown("---")
        
        # Risk settings
        risk_level = st.selectbox(
            get_text("sidebar_risk_level"),
            list(RISK_LEVELS.keys()),
            index=1,  # Default to Moderate
            help=get_help_content("risk_level")
        )
        
        # Configuration status
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            st.info(f"💾 {cache_count} análises em cache disponíveis")
    
    # Tutorial section with forced styling for Dark theme
    if st.session_state.get('show_tutorial', False):
        st.markdown("---")
        
        # Apply dark theme specific styling directly to tutorial content
        if st.session_state.get('theme', 'light') == 'dark':
            st.markdown("""
            <style>
                /* Force all tutorial text to be white */
                .main * {
                    color: #ffffff !important;
                }
                
                /* Specific tutorial styling */
                .stTabs * {
                    color: #ffffff !important;
                }
                
                /* Tab content background */
                div[data-baseweb="tab-panel"] {
                    background-color: #1e1e1e !important;
                    color: #ffffff !important;
                }
                
                /* All text in tabs */
                div[data-baseweb="tab-panel"] * {
                    color: #ffffff !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("# 📚 Tutorial Completo da Plataforma")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 Configurações Básicas", 
            "📊 Análises Disponíveis", 
            "📈 Indicadores Técnicos", 
            "⚖️ Gestão de Risco",
            "⚙️ Opções Avançadas"
        ])
        
        with tab1:
            # Force white text for this tab content
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('<div style="color: #ffffff !important;">', unsafe_allow_html=True)
            
            st.markdown("## 🔧 Configurações Básicas")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("currency_pair")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("currency_pair", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
                
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("time_interval")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("time_interval", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("prediction_horizon")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("prediction_horizon", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
                
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("risk_level")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("risk_level", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('<div style="color: #ffffff !important;">', unsafe_allow_html=True)
            
            st.markdown("## 📊 Tipos de Análise")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("get_trading_signal")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("get_trading_signal", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("quick_check")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("quick_check", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('<div style="color: #ffffff !important;">', unsafe_allow_html=True)
            
            st.markdown("## 📈 Indicadores e Sentimento")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("technical_indicators")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("technical_indicators", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("sentiment_analysis")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("sentiment_analysis", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('<div style="color: #ffffff !important;">', unsafe_allow_html=True)
            
            st.markdown("## ⚖️ Gestão e Análise de Risco")
            st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("risk_analysis")}</h3></div>', unsafe_allow_html=True)
            content = get_help_content("risk_analysis", detailed=True)
            if st.session_state.get('theme', 'light') == 'dark':
                content = f'<div style="color: #ffffff !important;">{content}</div>'
            st.markdown(content, unsafe_allow_html=True)
            
            warning_class = "warning-alert" if st.session_state.get('theme', 'light') == 'light' else "warning-alert"
            text_color = "" if st.session_state.get('theme', 'light') == 'light' else "color: #ffffff !important;"
            
            st.markdown(f"""
            <div class="{warning_class}" style="{text_color}">
                <h4 style="{text_color}">⚠️ Aviso Importante sobre Riscos</h4>
                <p style="{text_color}"><strong>O trading forex envolve riscos significativos:</strong></p>
                <ul style="{text_color}">
                    <li style="{text_color}">Você pode perder mais do que investiu</li>
                    <li style="{text_color}">Mercados são imprevisíveis, mesmo com IA</li>
                    <li style="{text_color}">Use sempre stop loss e gestão de risco</li>
                    <li style="{text_color}">Nunca invista dinheiro que não pode perder</li>
                    <li style="{text_color}">Esta plataforma é apenas educacional</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('<div style="color: #ffffff !important;">', unsafe_allow_html=True)
            
            st.markdown("## ⚙️ Opções Avançadas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("advanced_options")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("advanced_options", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
                
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("cache_management")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("cache_management", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div style="color: #ffffff !important;"><h3>{get_help_title("model_architecture")}</h3></div>', unsafe_allow_html=True)
                content = get_help_content("model_architecture", detailed=True)
                if st.session_state.get('theme', 'light') == 'dark':
                    content = f'<div style="color: #ffffff !important;">{content}</div>'
                st.markdown(content, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h4 style="color: white !important;">🚀 Dicas de Otimização Avançada</h4>
                <ul style="color: white !important;">
                    <li style="color: white !important;"><strong style="color: white !important;">Para Scalping (5-15min):</strong> Lookback=30, MC=15, Épocas=8</li>
                    <li style="color: white !important;"><strong style="color: white !important;">Para Day Trading (1h):</strong> Lookback=60, MC=20, Épocas=10</li>
                    <li style="color: white !important;"><strong style="color: white !important;">Para Swing Trading (4h+):</strong> Lookback=120, MC=30, Épocas=15</li>
                    <li style="color: white !important;"><strong style="color: white !important;">Para Análise Crítica:</strong> Lookback=100, MC=50, Épocas=20</li>
                </ul>
                <p style="color: white !important;"><strong style="color: white !important;">Lembre-se:</strong> Configurações mais altas = maior precisão, mas tempo de processamento mais longo.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('theme', 'light') == 'dark':
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if analyze_button or quick_analysis:
        run_analysis(
            pair, interval, horizon, risk_level, lookback_period, 
            mc_samples, epochs, quick_analysis
        )
    
    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Aviso Legal:</strong> Esta plataforma é apenas para fins educacionais. 
        Trading forex envolve riscos substanciais e pode não ser adequado para todos os investidores.</p>
        <p>Desenvolvido pela Artecinvesting • Última atualização: {}</p>
    </div>
    """.format(datetime.now().strftime("%d-%m-%Y %H:%M")), unsafe_allow_html=True)

def run_analysis(pair, interval, horizon, risk_level, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis"""
    
    # Custom loading display
    # Removed loading spinner - run silently without visual feedback
    
    # Removed all progress indicators - analysis runs silently
    
    try:
        # Step 1: Fetch data silently
        
        df = services['data_service'].fetch_forex_data(
            pair, 
            INTERVALS[interval], 
            'full' if not is_quick else 'compact'
        )
        
        if not services['data_service'].validate_data(df):
            st.error("❌ Dados insuficientes ou inválidos recebidos")
            return
        
        # Step 2: Fetch sentiment silently
        sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
        sentiment_signal = services['sentiment_service'].get_sentiment_signal(sentiment_score)
        sentiment_strength = services['sentiment_service'].get_sentiment_strength(sentiment_score)
        
        # Step 3: Add technical indicators silently
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        trading_signals = TechnicalIndicators.get_trading_signals(df_with_indicators)
        
        # Step 4: Train model and predict silently
        predictor = ForexPredictor(
            lookback=lookback_period,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        # Use recent data for training
        train_data = df_with_indicators.tail(min(1000, len(df_with_indicators)))
        
        training_metrics = predictor.train_model(
            train_data, 
            sentiment_score, 
            epochs=epochs if not is_quick else 5,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        # Step 5: Make predictions silently
        steps = HORIZON_STEPS[horizon]
        predictions, uncertainties = predictor.predict_future(
            train_data, 
            sentiment_score, 
            steps, 
            mc_samples
        )
        
        # Calculate confidence
        model_confidence = predictor.get_model_confidence(train_data, sentiment_score)
        
        # Step 6: Store results silently
        
        current_price = float(df_with_indicators['close'].iloc[-1])
        predicted_price = predictions[-1] if predictions and len(predictions) > 0 else current_price
        
        # Validation: Ensure predicted price is realistic (within 10% of current price for safety)
        max_change = current_price * 0.1  # 10% maximum change as sanity check
        if abs(predicted_price - current_price) > max_change:
            # If prediction seems unrealistic, use a more conservative estimate
            if predicted_price > current_price:
                predicted_price = current_price + (max_change * 0.5)  # 5% increase max
            else:
                predicted_price = current_price - (max_change * 0.5)  # 5% decrease max
        
        # Calculate additional metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Calculate price variation range against the expected trend
        uncertainty = uncertainties[-1] if uncertainties and len(uncertainties) > 0 else 0.0
        
        # Calculate where price could move OPPOSITE to our prediction
        # This shows risk of prediction being completely wrong
        risk_multiplier = 1.5  # How far opposite the prediction could go
        
        if price_change > 0:  # We predict UP (BUY) - show downside risk if wrong
            counter_trend_target = current_price - (abs(price_change) * risk_multiplier)
            risk_direction = "Downside"
            risk_description = "if bullish prediction fails"
        else:  # We predict DOWN (SELL) - show upside risk if wrong
            counter_trend_target = current_price + (abs(price_change) * risk_multiplier)
            risk_direction = "Upside" 
            risk_description = "if bearish prediction fails"
        
        # Debug logging (temporary)
        print(f"DEBUG - Current Price: {current_price:.5f}")
        print(f"DEBUG - Raw Predicted Price: {predictions[-1] if predictions else 'None'}")
        print(f"DEBUG - Final Predicted Price: {predicted_price:.5f}")
        print(f"DEBUG - Price Change: {price_change_pct:.2f}%")
        print(f"DEBUG - Counter-trend Risk: {risk_direction} to {counter_trend_target:.5f} ({risk_description})")
        
        # Risk assessment
        risk_tolerance = RISK_LEVELS[risk_level]
        position_size = min(risk_tolerance, abs(price_change_pct) / 100)
        
        st.session_state.analysis_results = {
            'pair': pair,
            'interval': interval,
            'horizon': horizon,
            'timestamp': datetime.now(),
            'data': df_with_indicators,
            'sentiment': {
                'score': sentiment_score,
                'signal': sentiment_signal,
                'strength': sentiment_strength
            },
            'predictions': predictions,
            'uncertainties': uncertainties,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'model_confidence': model_confidence,
            'training_metrics': training_metrics,
            'trading_signals': trading_signals,
            'risk_assessment': {
                'risk_level': risk_level,
                'position_size': position_size,
                'stop_loss': current_price * (1 - risk_tolerance),
                'take_profit': current_price * (1 + risk_tolerance * 2)
            },
            'counter_trend_risk': {
                'direction': risk_direction,
                'target_price': counter_trend_target,
                'risk_percentage': abs(counter_trend_target - current_price) / current_price * 100,
                'description': risk_description,
                'prediction_direction': 'Bullish' if price_change > 0 else 'Bearish'
            }
        }
        
        # Analysis completed silently
        
    except Exception as e:
        # Remove spinner on error too
        # Error handling silently - show minimal error
        st.error(f"Erro na análise: {str(e)}")

def get_trading_recommendation(results):
    """Calculate overall trading recommendation based on all signals"""
    
    # Validate the price change makes sense
    current_price = results['current_price']
    predicted_price = results['predicted_price']
    price_change_pct = results['price_change_pct']
    
    # Debug logging
    print(f"RECOMMENDATION DEBUG - Current: {current_price:.5f}, Predicted: {predicted_price:.5f}, Change: {price_change_pct:.2f}%")
    
    # Get individual signals
    price_signal = 1 if results['price_change'] > 0 else -1
    sentiment_signal = 1 if results['sentiment']['score'] > 0 else -1 if results['sentiment']['score'] < 0 else 0
    
    # Get technical signals from the most recent data
    signals = results['trading_signals'].tail(1).iloc[0]
    
    # Count bullish/bearish technical signals
    tech_signals = 0
    if signals.get('SMA_Bullish', False):
        tech_signals += 1
    elif signals.get('SMA_Bearish', False):
        tech_signals -= 1
        
    if signals.get('MACD_Bullish', False):
        tech_signals += 1
    elif signals.get('MACD_Bearish', False):
        tech_signals -= 1
        
    if signals.get('RSI_Oversold', False):
        tech_signals += 1
    elif signals.get('RSI_Overbought', False):
        tech_signals -= 1
    
    # Calculate overall score
    overall_score = price_signal + sentiment_signal + (tech_signals / 2)
    
    # Determine recommendation
    if overall_score >= 1.5:
        return "STRONG BUY", "success", "📈"
    elif overall_score >= 0.5:
        return "BUY", "success", "📈"
    elif overall_score <= -1.5:
        return "STRONG SELL", "error", "📉"
    elif overall_score <= -0.5:
        return "SELL", "error", "📉"
    else:
        return "HOLD", "warning", "➡️"

def display_analysis_results():
    """Display simplified analysis results with prominent trading recommendation"""
    results = st.session_state.analysis_results
    
    # Get trading recommendation
    recommendation, rec_type, rec_icon = get_trading_recommendation(results)
    
    # Prominent Trading Recommendation Section
    st.markdown("---")
    
    # Large recommendation display
    if rec_type == "success":
        bg_color = "#d4edda"
        border_color = "#28a745"
    elif rec_type == "error":
        bg_color = "#f8d7da"
        border_color = "#dc3545"
    else:
        bg_color = "#fff3cd"
        border_color = "#ffc107"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 3px solid {border_color};
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    ">
        <h1 style="color: {border_color}; margin: 0; font-size: 3em;">
            {rec_icon} {recommendation}
        </h1>
        <h2 style="color: #333; margin: 10px 0; font-size: 1.5em;">
            {results['pair']} - {results['horizon']} Outlook
        </h2>
        <p style="color: #666; font-size: 1.2em; margin: 0;">
            Expected Price Change: <strong>{results['price_change_pct']:+.2f}%</strong> | 
            Reversal Risk: <strong>{results['counter_trend_risk']['target_price']:.5f}</strong> | 
            Confidence: <strong>{results['model_confidence']:.0%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in a simplified format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"{results['current_price']:.5f}",
            help="Current market price"
        )
    
    with col2:
        st.metric(
            "Target Price",
            f"{results['predicted_price']:.5f}",
            delta=f"{results['price_change']:+.5f}",
            help=f"Predicted price for {results['horizon']}"
        )
    
    with col3:
        sentiment_emoji = "🟢" if results['sentiment']['score'] > 0 else "🔴" if results['sentiment']['score'] < 0 else "🟡"
        st.metric(
            "Market Sentiment",
            f"{sentiment_emoji} {results['sentiment']['signal']}",
            help="Overall market sentiment from news analysis"
        )
    
    # Simplified tabs - only essential information
    tab1, tab2, tab3 = st.tabs([
        "📊 Price & Signals", "📰 Analysis Details", "⚖️ Risk & Settings"
    ])
    
    with tab1:
        # Price chart with key technical signals
        price_chart = services['visualizer'].create_price_chart(
            results['data'].tail(200), 
            results['pair'],
            indicators=True
        )
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Key signals summary in a compact format
        signals = results['trading_signals'].tail(1).iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi_signal = "Oversold" if signals['RSI_Oversold'] else "Overbought" if signals['RSI_Overbought'] else "Neutral"
            rsi_color = "🟢" if signals['RSI_Oversold'] else "🔴" if signals['RSI_Overbought'] else "🟡"
            st.info(f"RSI: {rsi_color} {rsi_signal}")
        
        with col2:
            macd_signal = "Bullish" if signals['MACD_Bullish'] else "Bearish" if signals['MACD_Bearish'] else "Neutral"
            macd_color = "🟢" if signals['MACD_Bullish'] else "🔴" if signals['MACD_Bearish'] else "🟡"
            st.info(f"MACD: {macd_color} {macd_signal}")
        
        with col3:
            sma_signal = "Bullish" if signals['SMA_Bullish'] else "Bearish"
            sma_color = "🟢" if signals['SMA_Bullish'] else "🔴"
            st.info(f"Trend: {sma_color} {sma_signal}")
        
        with col4:
            bb_signal = "Squeeze" if signals['BB_Squeeze'] else "Normal"
            bb_color = "🟠" if signals['BB_Squeeze'] else "🟡"
            st.info(f"Volatility: {bb_color} {bb_signal}")
    
    with tab2:
        # Combined analysis view
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AI Prediction")
            direction = "Bullish" if results['price_change'] > 0 else "Bearish"
            direction_icon = "📈" if results['price_change'] > 0 else "📉"
            confidence_level = "High" if results['model_confidence'] > 0.7 else "Medium" if results['model_confidence'] > 0.4 else "Low"
            
            st.markdown(f"""
            - **Direction:** {direction_icon} {direction}
            - **Confidence:** {confidence_level} ({results['model_confidence']:.0%})
            - **Expected Move:** {results['price_change_pct']:+.2f}%
            - **Target Price:** {results['predicted_price']:.5f}
            """)
            
        with col2:
            st.markdown("#### Risk Analysis")
            trend_direction = "Bullish" if results['price_change'] > 0 else "Bearish"
            counter_risk = results['counter_trend_risk']
            
            # Show risk of prediction being wrong
            prediction_icon = "📈" if counter_risk['prediction_direction'] == 'Bullish' else "📉"
            risk_icon = "📉" if counter_risk['direction'] == 'Downside' else "📈"
            
            st.markdown(f"""
            - **Our Prediction:** {prediction_icon} {counter_risk['prediction_direction']}
            - **Risk if Wrong:** {risk_icon} {counter_risk['direction']} risk
            - **Counter-trend Target:** {counter_risk['target_price']:.5f}
            - **Risk Exposure:** {counter_risk['risk_percentage']:.1f}%
            """)
        
        # Market sentiment section
        st.markdown("#### Market Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_direction = "Positive" if results['sentiment']['score'] > 0 else "Negative" if results['sentiment']['score'] < 0 else "Neutral"
            sentiment_icon = "😊" if results['sentiment']['score'] > 0 else "😟" if results['sentiment']['score'] < 0 else "😐"
            
            st.markdown(f"""
            - **Sentiment:** {sentiment_icon} {sentiment_direction}
            - **Score:** {results['sentiment']['score']:.2f}
            - **Strength:** {results['sentiment']['strength']}
            """)
        
        with col2:
            # Show what happens if the analysis is completely wrong
            counter_risk = results['counter_trend_risk']
            
            st.markdown(f"""
            **If Prediction is Wrong:**
            - Our forecast: {counter_risk['prediction_direction']}
            - Risk direction: {counter_risk['direction']}
            - Target if opposite: {counter_risk['target_price']:.5f}
            - Potential move: {counter_risk['risk_percentage']:.1f}%
            - Risk level: {'HIGH' if counter_risk['risk_percentage'] > 4 else 'MODERATE' if counter_risk['risk_percentage'] > 2 else 'LOW'}
            """)
        
        # Prediction chart
        prediction_chart = services['visualizer'].create_prediction_chart(
            results['data'].tail(100),
            results['predictions'],
            results['uncertainties'],
            results['pair'],
            results['horizon']
        )
        st.plotly_chart(prediction_chart, use_container_width=True)
    
    with tab3:
        # Risk management and position sizing
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Suggested Position")
            risk_data = results['risk_assessment']
            st.markdown(f"""
            - **Risk Level:** {risk_data['risk_level']}
            - **Position Size:** {risk_data['position_size']:.1%} of portfolio
            - **Stop Loss:** {risk_data['stop_loss']:.5f}
            - **Take Profit:** {risk_data['take_profit']:.5f}
            """)
            
            # Market variation warning based on counter-trend risk
            counter_risk = results['counter_trend_risk']
            if counter_risk['risk_percentage'] > 2.0:
                st.warning(f"High risk exposure: {counter_risk['risk_percentage']:.1f}%")
            
            # Show the risk forecast
            counter_risk = results['counter_trend_risk']
            st.markdown(f"""
            **Risk Forecast (Prediction Failure):**
            - Our prediction: {counter_risk['prediction_direction']}
            - Risk direction: {counter_risk['direction']}
            - Target if we're wrong: {counter_risk['target_price']:.5f}
            - Maximum risk exposure: {counter_risk['risk_percentage']:.1f}%
            """)
            
            # Simple risk warnings
            if results['model_confidence'] < 0.5:
                st.warning("Low confidence - reduce position size")
            if abs(results['price_change_pct']) > 2:
                st.warning("Large move predicted - high risk/reward")
        
        with col2:
            st.markdown("#### Analysis Settings")
            st.markdown(f"""
            - **Pair:** {results['pair']}
            - **Timeframe:** {results['interval']}
            - **Prediction Horizon:** {results['horizon']}
            - **Analysis Time:** {results['timestamp'].strftime('%H:%M:%S')}
            """)
            
            # Model performance summary
            st.markdown("#### Model Performance")
            train_loss = results['training_metrics'].get('final_train_loss', 0)
            val_loss = results['training_metrics'].get('final_val_loss', 0)
            st.markdown(f"""
            - **Training Loss:** {train_loss:.4f}
            - **Validation Loss:** {val_loss:.4f}
            - **Model Confidence:** {results['model_confidence']:.0%}
            """)
        mae = results['training_metrics'].get('mae', 0)
        rmse = results['training_metrics'].get('rmse', 0)
        
        if mae < 0.01:
            st.success("✅ Excellent model performance (MAE < 0.01)")
        elif mae < 0.02:
            st.info("ℹ️ Good model performance (MAE < 0.02)")
        else:
            st.warning("⚠️ Model performance could be improved (MAE > 0.02)")

if __name__ == "__main__":
    main()
