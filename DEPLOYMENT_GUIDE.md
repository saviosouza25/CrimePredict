# Guia de Deploy no Streamlit Community Cloud

## Passo 1: Criar Conta no GitHub
1. Acesse github.com e crie uma conta gratuita
2. Crie um novo repositório público chamado "forex-analysis-platform"

## Passo 2: Preparar Arquivos
Você precisa dos seguintes arquivos no repositório:

### requirements.txt (renomear streamlit_requirements.txt)
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
matplotlib>=3.7.0
requests>=2.31.0
scikit-learn>=1.3.0
torch>=2.0.0
pytz>=2023.3
vadersentiment>=3.3.2
```

### .streamlit/config.toml
```
[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "dark"
```

### .streamlit/secrets.toml (para suas chaves de API)
```
ALPHA_VANTAGE_API_KEY = "sua_chave_aqui"
```

## Passo 3: Upload dos Arquivos
1. Copie app.py para o repositório
2. Copie todas as pastas (config, services, utils, models, etc)
3. Renomeie streamlit_requirements.txt para requirements.txt
4. Crie a pasta .streamlit com os arquivos de configuração

## Passo 4: Deploy no Streamlit Community Cloud
1. Acesse share.streamlit.io
2. Faça login com sua conta GitHub
3. Clique "New app"
4. Selecione seu repositório "forex-analysis-platform"
5. Main file path: app.py
6. Clique "Deploy!"

## Passo 5: Configurar Secrets
1. No painel do Streamlit Cloud, vá em "Settings" → "Secrets"
2. Adicione suas chaves de API:
```
ALPHA_VANTAGE_API_KEY = "sua_chave_aqui"
```

## Limitações do Plano Gratuito
- 1GB de RAM
- Repositório deve ser público
- App "hiberna" após 7 dias sem uso
- Reativação automática quando acessado

## Vantagens
- 100% gratuito
- Deploy automático a cada commit
- Domínio próprio (.streamlit.app)
- Suporte completo ao Streamlit