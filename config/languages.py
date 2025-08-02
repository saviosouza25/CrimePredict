"""
Language configuration for the Forex Analysis Platform
"""

# Brazilian Portuguese translations
PORTUGUESE_BR = {
    # Authentication
    "login_title": "Plataforma Avançada de Análise Forex",
    "login_subtitle": "Por favor, insira suas credenciais para acessar a plataforma",
    "login_secure_text": "Acesso seguro às ferramentas profissionais de análise forex",
    "username_placeholder": "Digite o nome de usuário",
    "password_placeholder": "Digite a senha",
    "invalid_credentials": "😞 Nome de usuário ou senha inválidos",
    
    # Main header
    "main_title": "🏢 Plataforma Avançada de Análise Forex",
    
    # Sidebar
    "sidebar_currency_pair": "Par de Moedas",
    "sidebar_time_interval": "Intervalo de Tempo",
    "sidebar_risk_level": "Nível de Risco",
    "sidebar_conservative": "Conservador",
    "sidebar_moderate": "Moderado",
    "sidebar_aggressive": "Agressivo",
    "sidebar_analyze_button": "🔍 Analisar",
    
    # Time intervals
    "1h": "1 Hora",
    "4h": "4 Horas", 
    "1d": "1 Dia",
    "1wk": "1 Semana",
    
    # Analysis sections
    "tab_overview": "Visão Geral",
    "tab_technical": "Análise Técnica",
    "tab_prediction": "Previsão",
    "tab_news": "Notícias",
    
    # Recommendation section
    "recommendation_title": "💡 Recomendação de Trading",
    "recommendation_buy": "COMPRAR",
    "recommendation_sell": "VENDER", 
    "recommendation_hold": "MANTER",
    
    # Current price
    "current_price": "Preço Atual",
    "price_change": "Variação 24h",
    "trend": "Tendência",
    "confidence": "Confiança",
    
    # Technical indicators
    "technical_indicators": "Indicadores Técnicos",
    "sma_20": "MMS 20",
    "sma_50": "MMS 50", 
    "rsi": "IFR",
    "macd": "MACD",
    "bollinger_bands": "Bandas de Bollinger",
    "stochastic": "Estocástico",
    
    # Signal descriptions
    "signal_buy": "Sinais indicam tendência de alta",
    "signal_sell": "Sinais indicam tendência de baixa",
    "signal_hold": "Sinais mistos - aguardar",
    
    # LSTM Prediction
    "lstm_prediction": "Previsão LSTM",
    "predicted_price": "Preço Previsto",
    "price_target": "Meta de Preço",
    "risk_analysis": "Análise de Risco",
    "risk_forecast": "Previsão de Risco",
    
    # Risk levels
    "upward_risk": "Risco de Alta",
    "downward_risk": "Risco de Baixa",
    "volatility": "Volatilidade",
    
    # Sentiment analysis
    "sentiment_analysis": "Análise de Sentimento",
    "market_sentiment": "Sentimento do Mercado",
    "positive": "Positivo",
    "negative": "Negativo",
    "neutral": "Neutro",
    "news_sentiment": "Sentimento das Notícias",
    
    # Chart titles
    "price_chart": "Gráfico de Preços",
    "technical_analysis": "Análise Técnica",
    "price_prediction": "Previsão de Preços",
    "prediction_start": "Início da Previsão",
    
    # Status messages
    "loading": "Carregando...",
    "analyzing": "Analisando dados...",
    "error_data": "Erro ao carregar dados. Tente novamente.",
    "error_api": "Erro na API. Verifique sua conexão.",
    "success_analysis": "Análise concluída com sucesso!",
    
    # Time labels
    "date": "Data",
    "time": "Hora",
    "price": "Preço",
    "volume": "Volume",
    "open": "Abertura",
    "high": "Máxima",
    "low": "Mínima", 
    "close": "Fechamento",
    
    # Prediction horizons
    "next_hour": "Próxima Hora",
    "next_4_hours": "Próximas 4 Horas",
    "next_day": "Próximo Dia",
    "next_week": "Próxima Semana",
    
    # Confidence levels
    "very_high": "Muito Alta",
    "high": "Alta",
    "medium": "Média",
    "low": "Baixa",
    "very_low": "Muito Baixa",
    
    # Additional elements
    "historical": "Histórico",
    "prediction": "Previsão",
    "upper_bound": "Limite Superior (95%)",
    "lower_bound": "Limite Inferior (95%)",
    "signal": "Sinal",
    "histogram": "Histograma",
    
    # Navigation
    "back": "Voltar",
    "next": "Próximo",
    "refresh": "Atualizar",
    "export": "Exportar",
    "print": "Imprimir",
    
    # Error messages
    "connection_error": "Erro de conexão. Verifique sua internet.",
    "invalid_pair": "Par de moedas inválido.",
    "no_data": "Nenhum dado disponível para o período selecionado.",
    "analysis_failed": "Falha na análise. Tente novamente mais tarde."
}

# Function to get translated text
def get_text(key, language="pt-br"):
    """Get translated text for given key"""
    if language == "pt-br":
        return PORTUGUESE_BR.get(key, key)
    return key  # Fallback to key if translation not found