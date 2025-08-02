"""
Help content and tutorials for the Forex Analysis Platform
"""

HELP_CONTENT = {
    "currency_pair": {
        "title": "Par de Moedas",
        "short": "Selecione o par de moedas para análise",
        "detailed": """
        <h4>🌍 Par de Moedas - O Que É?</h4>
        <p>Um par de moedas mostra a relação de valor entre duas moedas diferentes.</p>
        
        <h5>📊 Como Funciona:</h5>
        <ul>
            <li><strong>EUR/USD:</strong> Euro vs Dólar Americano</li>
            <li><strong>GBP/JPY:</strong> Libra Esterlina vs Iene Japonês</li>
            <li><strong>USD/BRL:</strong> Dólar vs Real Brasileiro</li>
        </ul>
        
        <h5>💡 Exemplo Prático:</h5>
        <p>Se EUR/USD = 1.2000, significa que 1 Euro vale 1.20 Dólares.</p>
        
        <h5>🎯 Principais Pares:</h5>
        <ul>
            <li><strong>Majors:</strong> EUR/USD, GBP/USD, USD/JPY (mais líquidos)</li>
            <li><strong>Minors:</strong> EUR/GBP, AUD/CAD (menos volume)</li>
            <li><strong>Exóticos:</strong> USD/ZAR, EUR/TRY (mais voláteis)</li>
        </ul>
        """
    },
    
    "time_interval": {
        "title": "Intervalo de Tempo",
        "short": "Período do gráfico para análise",
        "detailed": """
        <h4>⏰ Intervalo de Tempo - Escolha Seu Horizonte</h4>
        <p>Define o período de cada vela (candlestick) no gráfico de preços.</p>
        
        <h5>📈 Tipos de Intervalos:</h5>
        <ul>
            <li><strong>1 Hora (1h):</strong> Cada vela = 1 hora de trading</li>
            <li><strong>4 Horas (4h):</strong> Cada vela = 4 horas de dados</li>
            <li><strong>1 Dia (1d):</strong> Cada vela = 1 dia completo</li>
            <li><strong>1 Semana (1w):</strong> Cada vela = 1 semana de trading</li>
        </ul>
        
        <h5>🎯 Quando Usar Cada Um:</h5>
        <ul>
            <li><strong>1h:</strong> Day trading, decisões rápidas</li>
            <li><strong>4h:</strong> Swing trading, análise intraday</li>
            <li><strong>1d:</strong> Posições de médio prazo</li>
            <li><strong>1w:</strong> Análise de tendência de longo prazo</li>
        </ul>
        
        <h5>💡 Dica:</h5>
        <p>Intervalos menores = mais sinais, mas mais ruído. Intervalos maiores = sinais mais confiáveis, mas menos frequentes.</p>
        """
    },
    
    "risk_level": {
        "title": "Nível de Risco",
        "short": "Sua tolerância ao risco para dimensionamento de posição",
        "detailed": """
        <h4>⚖️ Nível de Risco - Gerencie Seu Capital</h4>
        <p>Define quanto do seu capital você está disposto a arriscar em cada operação.</p>
        
        <h5>🛡️ Níveis Disponíveis:</h5>
        <ul>
            <li><strong>Conservador:</strong> 1-2% do capital por trade</li>
            <li><strong>Moderado:</strong> 2-5% do capital por trade</li>
            <li><strong>Agressivo:</strong> 5-10% do capital por trade</li>
        </ul>
        
        <h5>💰 Exemplo Prático:</h5>
        <p>Com capital de R$ 10.000:</p>
        <ul>
            <li><strong>Conservador:</strong> Risco máximo R$ 200 por trade</li>
            <li><strong>Moderado:</strong> Risco máximo R$ 500 por trade</li>
            <li><strong>Agressivo:</strong> Risco máximo R$ 1.000 por trade</li>
        </ul>
        
        <h5>⚠️ Regra de Ouro:</h5>
        <p>Nunca arrisque mais do que pode perder. O mercado forex é altamente volátil!</p>
        """
    },
    
    "get_trading_signal": {
        "title": "Obter Sinal de Trading",
        "short": "Executar análise completa com previsão IA",
        "detailed": """
        <h4>🎯 Sinal de Trading - Análise Completa com IA</h4>
        <p>Esta função executa uma análise abrangente combinando múltiplas tecnologias.</p>
        
        <h5>🔍 O Que a Análise Inclui:</h5>
        <ul>
            <li><strong>Dados Históricos:</strong> Busca preços históricos da API Alpha Vantage</li>
            <li><strong>Indicadores Técnicos:</strong> RSI, MACD, Médias Móveis, Bollinger Bands</li>
            <li><strong>Sentimento do Mercado:</strong> Análise de notícias financeiras</li>
            <li><strong>IA LSTM:</strong> Rede neural para previsão de preços</li>
        </ul>
        
        <h5>🧠 Como Funciona a IA:</h5>
        <ol>
            <li>Treina modelo LSTM com dados históricos</li>
            <li>Analisa padrões de preços passados</li>
            <li>Gera previsão de preço futuro</li>
            <li>Calcula incerteza da previsão</li>
        </ol>
        
        <h5>📊 Resultado Final:</h5>
        <ul>
            <li><strong>COMPRAR:</strong> Expectativa de alta no preço</li>
            <li><strong>VENDER:</strong> Expectativa de baixa no preço</li>
            <li><strong>MANTER:</strong> Sinais conflitantes ou mercado lateral</li>
        </ul>
        
        <h5>⏱️ Tempo de Processamento:</h5>
        <p>2-3 minutos para análise completa (treinamento da IA incluído)</p>
        """
    },
    
    "quick_check": {
        "title": "Verificação Rápida",
        "short": "Análise rápida com sinais básicos",
        "detailed": """
        <h4>⚡ Verificação Rápida - Análise Express</h4>
        <p>Versão simplificada da análise para decisões rápidas.</p>
        
        <h5>🔍 O Que Inclui:</h5>
        <ul>
            <li><strong>Indicadores Básicos:</strong> RSI, MACD, Médias Móveis</li>
            <li><strong>Sentimento Simplificado:</strong> Análise básica de notícias</li>
            <li><strong>IA Reduzida:</strong> Menos épocas de treinamento</li>
        </ul>
        
        <h5>⚡ Vantagens:</h5>
        <ul>
            <li>Resultado em 30-60 segundos</li>
            <li>Menos uso de recursos computacionais</li>
            <li>Ideal para verificações frequentes</li>
        </ul>
        
        <h5>🎯 Quando Usar:</h5>
        <ul>
            <li>Verificação rápida de tendência</li>
            <li>Confirmação de análise anterior</li>
            <li>Monitoramento de múltiplos pares</li>
        </ul>
        
        <h5>⚠️ Limitações:</h5>
        <p>Menos preciso que a análise completa. Use para orientação geral, não para decisões críticas.</p>
        """
    },
    
    "prediction_horizon": {
        "title": "Período de Previsão",
        "short": "Tempo de previsão",
        "detailed": """
        <h4>🔮 Período de Previsão - Veja o Futuro</h4>
        <p>Define por quanto tempo no futuro a IA tentará prever o movimento dos preços.</p>
        
        <h5>⏰ Horizontes Disponíveis:</h5>
        <ul>
            <li><strong>5 Minutos:</strong> Previsão ultra rápida para scalping</li>
            <li><strong>15 Minutos:</strong> Previsão de curto prazo para day trading</li>
            <li><strong>1 Hora:</strong> Previsão para 1 hora à frente</li>
            <li><strong>4 Horas:</strong> Previsão para 4 horas à frente</li>
            <li><strong>1 Dia:</strong> Previsão para 24 horas à frente</li>
            <li><strong>1 Semana:</strong> Previsão para 7 dias à frente</li>
        </ul>
        
        <h5>🎯 Precisão vs Horizonte:</h5>
        <ul>
            <li><strong>Ultra Curto (5-15min):</strong> Máxima precisão, ideal para scalping</li>
            <li><strong>Curto Prazo (1h):</strong> Alta precisão, menos incerteza</li>
            <li><strong>Médio Prazo (4h-1d):</strong> Boa precisão, equilibrio ideal</li>
            <li><strong>Longo Prazo (1w):</strong> Menor precisão, mais incerteza</li>
        </ul>
        
        <h5>💡 Estratégias por Horizonte:</h5>
        <ul>
            <li><strong>5min:</strong> Scalping rápido, múltiplas operações por hora</li>
            <li><strong>15min:</strong> Scalping e micro day trading</li>
            <li><strong>1h:</strong> Day trading clássico</li>
            <li><strong>4h:</strong> Swing trading intraday</li>
            <li><strong>1d:</strong> Position trading</li>
            <li><strong>1w:</strong> Investimento de tendência</li>
        </ul>
        """
    },
    
    "technical_indicators": {
        "title": "Indicadores Técnicos",
        "short": "Análise técnica dos preços",
        "detailed": """
        <h4>📊 Indicadores Técnicos - Decifrando os Gráficos</h4>
        <p>Ferramentas matemáticas que analisam padrões de preços para identificar tendências.</p>
        
        <h5>🔍 Principais Indicadores:</h5>
        
        <h6>📈 RSI (Índice de Força Relativa):</h6>
        <ul>
            <li>Mede se o ativo está sobrecomprado ou sobrevendido</li>
            <li>Escala de 0 a 100</li>
            <li>RSI > 70: Sobrecomprado (possível venda)</li>
            <li>RSI < 30: Sobrevendido (possível compra)</li>
        </ul>
        
        <h6>📊 MACD (Convergência e Divergência de Médias Móveis):</h6>
        <ul>
            <li>Identifica mudanças na força e direção da tendência</li>
            <li>Cruzamento acima da linha zero: Sinal de compra</li>
            <li>Cruzamento abaixo da linha zero: Sinal de venda</li>
        </ul>
        
        <h6>📉 Médias Móveis (SMA 20/50):</h6>
        <ul>
            <li>Suaviza flutuações de preços</li>
            <li>Preço acima da média: Tendência de alta</li>
            <li>Preço abaixo da média: Tendência de baixa</li>
            <li>Cruzamento de médias: Mudança de tendência</li>
        </ul>
        
        <h6>🎯 Bandas de Bollinger:</h6>
        <ul>
            <li>Mostram volatilidade e níveis de suporte/resistência</li>
            <li>Preço na banda superior: Possível reversão para baixo</li>
            <li>Preço na banda inferior: Possível reversão para cima</li>
        </ul>
        """
    },
    
    "sentiment_analysis": {
        "title": "Análise de Sentimento",
        "short": "Sentimento do mercado baseado em notícias",
        "detailed": """
        <h4>📰 Análise de Sentimento - O Humor do Mercado</h4>
        <p>Avalia o sentimento dos investidores através da análise de notícias financeiras.</p>
        
        <h5>🔍 Como Funciona:</h5>
        <ol>
            <li>Coleta notícias recentes sobre o par de moedas</li>
            <li>Analisa o tom das notícias (positivo/negativo/neutro)</li>
            <li>Gera uma pontuação de sentimento</li>
            <li>Combina com análise técnica para decisão final</li>
        </ol>
        
        <h5>📊 Interpretação dos Resultados:</h5>
        <ul>
            <li><strong>Positivo (0.1 a 1.0):</strong> Notícias otimistas, possível alta</li>
            <li><strong>Neutro (-0.1 a 0.1):</strong> Notícias equilibradas</li>
            <li><strong>Negativo (-1.0 a -0.1):</strong> Notícias pessimistas, possível baixa</li>
        </ul>
        
        <h5>💡 Exemplos de Impacto:</h5>
        <ul>
            <li><strong>Decisões do Banco Central:</strong> Aumento de juros = moeda forte</li>
            <li><strong>Dados Econômicos:</strong> PIB alto = sentimento positivo</li>
            <li><strong>Crises Políticas:</strong> Instabilidade = sentimento negativo</li>
            <li><strong>Acordos Comerciais:</strong> Novos acordos = otimismo</li>
        </ul>
        
        <h5>⚠️ Limitações:</h5>
        <ul>
            <li>Notícias podem ser manipuladas</li>
            <li>Mercado nem sempre reage racionalmente</li>
            <li>Eventos inesperados podem mudar tudo rapidamente</li>
        </ul>
        """
    },
    
    "risk_analysis": {
        "title": "Análise de Risco",
        "short": "Avaliação de riscos da operação",
        "detailed": """
        <h4>⚖️ Análise de Risco - Proteja Seu Capital</h4>
        <p>Calcula os riscos potenciais da operação e cenários adversos.</p>
        
        <h5>🎯 Tipos de Risco Analisados:</h5>
        
        <h6>📉 Risco Contra-Tendência:</h6>
        <ul>
            <li>Mostra onde o preço pode ir se a previsão estiver errada</li>
            <li>Se prevemos ALTA: mostra risco de QUEDA</li>
            <li>Se prevemos BAIXA: mostra risco de ALTA</li>
        </ul>
        
        <h6>📊 Volatilidade:</h6>
        <ul>
            <li>Mede o quanto o preço pode variar</li>
            <li>Alta volatilidade = maior risco e oportunidade</li>
            <li>Baixa volatilidade = movimentos mais previsíveis</li>
        </ul>
        
        <h5>💰 Cálculos de Posição:</h5>
        <ul>
            <li><strong>Stop Loss:</strong> Ponto de saída se perder</li>
            <li><strong>Take Profit:</strong> Ponto de saída se ganhar</li>
            <li><strong>Tamanho da Posição:</strong> Quanto investir</li>
            <li><strong>Razão Risco/Retorno:</strong> Quanto ganhar vs perder</li>
        </ul>
        
        <h5>📈 Exemplo Prático:</h5>
        <p>EUR/USD = 1.2000, Previsão = 1.2100 (COMPRAR)</p>
        <ul>
            <li><strong>Lucro Esperado:</strong> +100 pips</li>
            <li><strong>Stop Loss:</strong> 1.1950 (-50 pips)</li>
            <li><strong>Risco Contra-Tendência:</strong> 1.1850 (se análise falhar)</li>
            <li><strong>Razão R/R:</strong> 2:1 (ganhar R$ 200 vs perder R$ 100)</li>
        </ul>
        """
    }
}

def get_help_content(key, detailed=False):
    """Get help content for a specific function"""
    content = HELP_CONTENT.get(key, {})
    if detailed:
        return content.get("detailed", "Ajuda não disponível para esta função.")
    return content.get("short", "Selecionar opção")

def get_help_title(key):
    """Get help title for a specific function"""
    return HELP_CONTENT.get(key, {}).get("title", key)