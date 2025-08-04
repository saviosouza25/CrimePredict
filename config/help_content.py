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
            <li><strong>5 Minutos:</strong> Previsão ultra rápida para operações de muito curto prazo</li>
            <li><strong>15 Minutos:</strong> Previsão de curto prazo para day trading</li>
            <li><strong>1 Hora:</strong> Previsão para 1 hora à frente</li>
            <li><strong>4 Horas:</strong> Previsão para 4 horas à frente</li>
            <li><strong>1 Dia:</strong> Previsão para 24 horas à frente</li>
            <li><strong>1 Semana:</strong> Previsão para 7 dias à frente</li>
        </ul>
        
        <h5>🎯 Precisão vs Horizonte:</h5>
        <ul>
            <li><strong>Ultra Curto (5-15min):</strong> Máxima precisão, ideal para operações rápidas</li>
            <li><strong>Curto Prazo (1h):</strong> Alta precisão, menos incerteza</li>
            <li><strong>Médio Prazo (4h-1d):</strong> Boa precisão, equilibrio ideal</li>
            <li><strong>Longo Prazo (1w):</strong> Menor precisão, mais incerteza</li>
        </ul>
        
        <h5>💡 Estratégias por Horizonte:</h5>
        <ul>
            <li><strong>5min:</strong> Operações ultra rápidas, múltiplas por hora</li>
            <li><strong>15min:</strong> Micro day trading e operações rápidas</li>
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
    },
    
    "advanced_options": {
        "title": "Opções Avançadas",
        "short": "Configurações avançadas da plataforma",
        "detailed": """
        <h4>⚙️ Opções Avançadas - Controle Total da Análise</h4>
        <p>Estas configurações permitem personalizar profundamente o comportamento da IA e dos algoritmos de análise.</p>
        
        <h5>🧠 Parâmetros de Machine Learning:</h5>
        <ul>
            <li><strong>Lookback Period (Período de Observação):</strong> Quantos pontos históricos a IA analisa</li>
            <li><strong>Monte Carlo Samples:</strong> Número de simulações para calcular incerteza</li>
            <li><strong>Épocas de Treinamento:</strong> Quantas vezes a IA treina com os dados</li>
        </ul>
        
        <h5>📊 Lookback Period (Padrão: 60):</h5>
        <ul>
            <li><strong>30-50:</strong> Análise mais rápida, foco em padrões recentes</li>
            <li><strong>60-100:</strong> Equilibrio entre velocidade e precisão (recomendado)</li>
            <li><strong>120-200:</strong> Análise mais lenta, considera mais histórico</li>
        </ul>
        
        <h5>🎲 Monte Carlo Samples (Padrão: 20):</h5>
        <ul>
            <li><strong>10-15:</strong> Estimativa rápida de incerteza</li>
            <li><strong>20-30:</strong> Boa precisão na incerteza (recomendado)</li>
            <li><strong>50-100:</strong> Máxima precisão, processamento mais lento</li>
        </ul>
        
        <h5>🔄 Épocas de Treinamento (Padrão: 10):</h5>
        <ul>
            <li><strong>5-8:</strong> Treinamento rápido, pode subajustar</li>
            <li><strong>10-15:</strong> Treinamento equilibrado (recomendado)</li>
            <li><strong>20-50:</strong> Treinamento intensivo, risco de sobreajuste</li>
        </ul>
        
        <h5>⚡ Impacto no Desempenho:</h5>
        <ul>
            <li><strong>Configuração Rápida:</strong> Lookback=30, MC=10, Épocas=5 (30s)</li>
            <li><strong>Configuração Padrão:</strong> Lookback=60, MC=20, Épocas=10 (2-3min)</li>
            <li><strong>Configuração Precisa:</strong> Lookback=120, MC=50, Épocas=20 (5-8min)</li>
        </ul>
        
        <h5>🎯 Quando Ajustar:</h5>
        <ul>
            <li><strong>Mercado Volátil:</strong> Reduzir lookback, aumentar MC samples</li>
            <li><strong>Mercado Estável:</strong> Aumentar lookback, manter MC padrão</li>
            <li><strong>Análise Rápida:</strong> Reduzir todos os parâmetros</li>
            <li><strong>Análise Crítica:</strong> Aumentar épocas e MC samples</li>
        </ul>
        """
    },
    
    "cache_management": {
        "title": "Gerenciamento de Cache",
        "short": "Sistema de cache da plataforma",
        "detailed": """
        <h4>💾 Gerenciamento de Cache - Otimização de Performance</h4>
        <p>O sistema de cache armazena análises e dados para acelerar operações futuras.</p>
        
        <h5>📁 Tipos de Cache:</h5>
        <ul>
            <li><strong>Cache de Dados:</strong> Preços históricos (15 min TTL)</li>
            <li><strong>Cache de Notícias:</strong> Sentimento de mercado (30 min TTL)</li>
            <li><strong>Cache de Análises:</strong> Resultados completos (5 min TTL)</li>
            <li><strong>Cache de Modelos:</strong> Modelos LSTM treinados (sessão)</li>
        </ul>
        
        <h5>🔄 Ciclo de Vida do Cache:</h5>
        <ol>
            <li><strong>Primeira Execução:</strong> Busca dados da API, treina modelo</li>
            <li><strong>Cache Hit:</strong> Usa dados armazenados, acelera análise</li>
            <li><strong>Expiração TTL:</strong> Cache expira, busca dados atualizados</li>
            <li><strong>Limpeza Manual:</strong> Usuário força atualização</li>
        </ol>
        
        <h5>⚡ Benefícios do Cache:</h5>
        <ul>
            <li><strong>Velocidade:</strong> Análises subsequentes 10x mais rápidas</li>
            <li><strong>Economia de API:</strong> Reduz chamadas desnecessárias</li>
            <li><strong>Consistência:</strong> Mesmos dados para comparações</li>
            <li><strong>Experiência:</strong> Interface mais responsiva</li>
        </ul>
        
        <h5>🗑️ Quando Limpar Cache:</h5>
        <ul>
            <li><strong>Dados Desatualizados:</strong> Após eventos de mercado importantes</li>
            <li><strong>Mudança de Estratégia:</strong> Quando alterar parâmetros avançados</li>
            <li><strong>Problemas de API:</strong> Se dados parecem incorretos</li>
            <li><strong>Nova Sessão:</strong> Para começar com dados frescos</li>
        </ul>
        
        <h5>📊 Indicadores de Cache:</h5>
        <ul>
            <li><strong>"X análises em cache":</strong> Quantidade de resultados armazenados</li>
            <li><strong>Velocidade de Carregamento:</strong> Cache ativo = carregamento rápido</li>
            <li><strong>Timestamp:</strong> Mostra quando dados foram atualizados</li>
        </ul>
        """
    },
    
    "model_architecture": {
        "title": "Arquitetura do Modelo",
        "short": "Detalhes técnicos da IA",
        "detailed": """
        <h4>🏗️ Arquitetura do Modelo LSTM - Tecnologia por Trás da IA</h4>
        <p>Nossa plataforma usa redes neurais LSTM (Long Short-Term Memory) especializadas em séries temporais.</p>
        
        <h5>🧠 Componentes da Rede Neural:</h5>
        <ul>
            <li><strong>Camadas LSTM:</strong> 2 camadas com 64 neurônios cada</li>
            <li><strong>Dropout:</strong> 30% para prevenir sobreajuste</li>
            <li><strong>Attention Mechanism:</strong> Foca nos padrões mais importantes</li>
            <li><strong>Dense Layer:</strong> Camada final para previsão de preços</li>
        </ul>
        
        <h5>📊 Entrada de Dados (Features):</h5>
        <ul>
            <li><strong>Preços OHLC:</strong> Open, High, Low, Close</li>
            <li><strong>Volume:</strong> Volume de negociação</li>
            <li><strong>Indicadores Técnicos:</strong> RSI, MACD, Médias Móveis</li>
            <li><strong>Sentimento:</strong> Score de análise de notícias</li>
            <li><strong>Volatilidade:</strong> Bandas de Bollinger e ATR</li>
        </ul>
        
        <h5>🔄 Processo de Treinamento:</h5>
        <ol>
            <li><strong>Normalização:</strong> MinMaxScaler padroniza os dados</li>
            <li><strong>Sequenciamento:</strong> Cria janelas temporais (lookback)</li>
            <li><strong>Treinamento:</strong> Adam optimizer com learning rate 0.001</li>
            <li><strong>Validação:</strong> Cross-validation temporal</li>
            <li><strong>Fine-tuning:</strong> Ajuste com dados mais recentes</li>
        </ol>
        
        <h5>🎯 Saídas do Modelo:</h5>
        <ul>
            <li><strong>Previsão de Preço:</strong> Valor futuro esperado</li>
            <li><strong>Intervalos de Confiança:</strong> Via Monte Carlo Dropout</li>
            <li><strong>Probabilidade de Direção:</strong> Chance de alta/baixa</li>
            <li><strong>Score de Confiança:</strong> Certeza do modelo (0-100%)</li>
        </ul>
        
        <h5>⚙️ Hiperparâmetros Técnicos:</h5>
        <ul>
            <li><strong>Hidden Size:</strong> 64 neurônios por camada LSTM</li>
            <li><strong>Layers:</strong> 2 camadas LSTM empilhadas</li>
            <li><strong>Dropout Rate:</strong> 0.3 (30%)</li>
            <li><strong>Learning Rate:</strong> 0.001 (Adam)</li>
            <li><strong>Batch Size:</strong> 32 amostras por batch</li>
        </ul>
        
        <h5>🔍 Validação e Métricas:</h5>
        <ul>
            <li><strong>MSE (Mean Squared Error):</strong> Erro quadrático médio</li>
            <li><strong>MAE (Mean Absolute Error):</strong> Erro absoluto médio</li>
            <li><strong>Directional Accuracy:</strong> % de acerto na direção</li>
            <li><strong>Sharpe Ratio:</strong> Retorno ajustado ao risco</li>
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