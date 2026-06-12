# Tech Challenge - Fase 1: Classificação de Risco de AVC
**Pós-Graduação em IA para DEVs 10IADT — FIAP**

Este repositório contém a solução desenvolvida para o **Tech Challenge B (Fase 1)**. O objetivo do projeto é construir um modelo preditivo robusto para identificar o risco de Acidente Vascular Cerebral (AVC) utilizando o *Stroke Prediction Dataset* do Kaggle, servindo como ferramenta de suporte à decisão clínica (CDSS). Além do desenvolvimento e análise do modelo via Jupyter Notebook, a solução inclui a produtização do modelo através de uma API REST desenvolvida com **FastAPI**, empacotada em um contêiner **Docker** e com um pipeline automatizado de CI/CD via **GitHub Actions** para publicação da imagem no GitHub Container Registry (GHCR).

---

## 📊 Base de Dados

O dataset utilizado neste projeto é o **Stroke Prediction Dataset**, disponível publicamente no Kaggle:
*   **Link para Download**: [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
*   **Arquivo Principal**: `healthcare-dataset-stroke-data.csv`

O dataset original possui 5.110 registros clínicos e 12 variáveis de dados contendo fatores de risco associados ao AVC.

---

## 📋 Estrutura do Projeto

O projeto é estruturado em um único notebook Jupyter contendo rigor metodológico, justificativas técnicas e discussões clínicas de viabilidade:

*   **`tech_challenge_fase_1.ipynb`**: Notebook Jupyter contendo todo o ciclo de vida do modelo:
    *   **EDA**: Separação visual com análises individuais (gráficos de pizza para hipertensão/cardiopatias, histogramas para categóricas, histogramas e boxplots para validação de outliers nas numéricas, e matriz de correlação ampliada).
    *   **Pré-processamento**: Divisão de dados e uso de pipelines (`ColumnTransformer`).
    *   **Modelagem**: Treinamento comparativo de 5 modelos de classificação clássicos.
    *   **Avaliação**: Avaliação focada em Recall, F1-Score, Matrizes de Confusão, Curva ROC e AUC-ROC.
    *   **Explicabilidade**: Análise via importância de atributos e coeficientes lineares.
    *   **Exportação**: Geração automática do arquivo de deploy.
*   **`dataset/`**: Fallback do dataset caso não seja possível baixar o Kaggle.
*   **`requirements.txt`**: Dependências do projeto.
*   **`Dockerfile`**: Imagem Docker contendo a API FastAPI e o modelo de ML.
*   **`.github/workflows/docker-publish.yml`**: Workflow do GitHub Actions para automação de CI/CD.

---

## 🛠️ Tecnologias e Dependências

As seguintes tecnologias são utilizadas no desenvolvimento deste projeto:

*   **Python 3.12**
*   **pandas** & **numpy**: Manipulação e análise estatística dos dados.
*   **scikit-learn**: Criação de pipelines robustos de preprocessamento e algoritmos de Machine Learning (Regressão Logística, Árvore de Decisão, Random Forest, KNN e SVM).
*   **shap**: Explicabilidade global e local das decisões do modelo baseada na Teoria dos Jogos (valores SHAP).
*   **matplotlib** & **seaborn**: Visualizações e gráficos (heatmaps, boxplots, gráficos de pizza e curvas ROC).
*   **kagglehub**: Carregamento automatizado e seguro do dataset direto do Kaggle.
*   **FastAPI** & **Uvicorn**: Construção e servidor da API RESTful para realizar as predições em tempo real.
*   **Docker**: Conteinerização da aplicação para garantir portabilidade e consistência entre ambientes.
*   **GitHub Actions**: Automação do pipeline de CI/CD para construir e publicar a imagem Docker.

---

## 🚀 Como Executar o Projeto Localmente

Siga os passos abaixo para preparar o ambiente e executar o notebook localmente.

### 1. Criar e Ativar o Ambiente Virtual
É necessário criar um ambiente virtual isolado para instalar as dependências do projeto. 

Para criar o ambiente virtual (com o nome `.venv`), execute o seguinte comando na raiz do projeto:
```bash
python -m venv .venv
```

Após a criação, você deve ativá-lo:

No Linux/macOS:
```bash
source .venv/bin/activate
```

No Windows (Prompt de Comando):
```cmd
.venv\\Scripts\\activate
```

No Windows (PowerShell):
```powershell
.venv\\Scripts\\Activate.ps1
```

### 2. Instalar as Dependências
Com o ambiente virtual ativo, instale as dependências necessárias executando:

```bash
pip install -r requirements.txt 
```

### 3. Baixar o Dataset
O dataset será baixado automaticamente pelo script. Certifique-se de ter uma conexão com a internet.

### 4. Abrir o Notebook Jupyter
Inicie a interface do Jupyter Notebook no seu terminal:
```bash
jupyter notebook
```
Ou, se preferir, abra a pasta do projeto no **Visual Studio Code** (VS Code) e utilize a extensão nativa de Jupyter Notebooks, selecionando o kernel da `.venv`.

### 4. Executar as Células
Abra o arquivo `tech_challenge_fase_1.ipynb` e execute as células sequencialmente (`Shift + Enter`).
*   **Nota**: O carregamento da base de dados será feito de forma transparente via API do `kagglehub` (sem necessidade de token manual) ou pelo link de fallback de repositório público configurado no notebook.

---

## 🔍 Visão Geral do Pipeline de ML

1.  **EDA (Análise Exploratória)**: Estatísticas das features, detecção de nulos na coluna `bmi` e visualização do desbalanceamento severo (~5% positivos vs ~95% negativos). Inclui análise individual e detalhada das distribuições das variáveis contínuas (com boxplots e histogramas) e categóricas (com gráficos de pizza para hipertensão/cardiopatias e histogramas para as demais).
2.  **Pré-processamento com Data Leakage Prevention**: Divisão em Treino/Validação/Teste antes de aplicar o `ColumnTransformer` (SimpleImputer por mediana para `bmi`, StandardScaler para numéricas e OneHotEncoder para categóricas).
3.  **Tratamento de Desbalanceamento**: Aplicação do parâmetro `class_weight='balanced'` diretamente nos classificadores clássicos para lidar com o desbalanceamento severo da base.
4.  **Avaliação Focada em Recall (Sensibilidade) e AUC-ROC**: Justificativa clínico-acadêmica de por que o Recall é a métrica mais crítica no cenário médico de triagem (reduzir a zero os falsos negativos), combinada com a análise comparativa de Curvas ROC e métricas AUC de cada modelo.
5.  **Explicabilidade do Modelo (SHAP)**: Análise comparativa utilizando o modelo de explicabilidade pós-hoc **SHAP** (SHapley Additive exPlanations) para interpretar o impacto individual de cada variável clínica nas previsões do Random Forest e da Regressão Logística, além das métricas nativas de feature importance e coeficientes.
6.  **Viabilidade Clínica**: Reflexão sobre as barreiras e uso prático do modelo integrado com Prontuários Eletrônicos (PEP) para triagem de urgência na "Hora de Ouro" do AVC.

---

## ⚙️ Exportando o Melhor Modelo

Para servir o modelo via API, o pipeline de treino do Random Forest é automaticamente exportado para o arquivo `best_model.joblib` na última célula do notebook `tech_challenge_fase_1.ipynb`:

```python
import joblib
# Exportando o pipeline completo do Random Forest (Considerado o melhor modelo para esta aplicação)
joblib.dump(rf_pipeline, 'best_model.joblib')
```

O arquivo `best_model.joblib` será salvo no diretório raiz do projeto e é lido pela API REST no momento da inicialização.

---

## 🐳 Como Executar a API via Docker (FastAPI)

Este projeto inclui uma API construída com **FastAPI** para realizar previsões em tempo real, que pode ser servida usando Docker.

### 1. Construir a Imagem Docker
Certifique-se de que o arquivo `best_model.joblib` foi gerado (ver seção acima). Em seguida, execute no terminal (dentro da pasta do projeto):

```bash
docker build -t stroke-prediction-api .
```

### 2. Executar o Contêiner
Inicie o contêiner mapeando a porta 8000:

```bash
docker run -d -p 8000:8000 stroke-prediction-api
```

A API estará disponível em `http://localhost:8000`.

### 3. Testar a API

Você pode testar a API através da documentação interativa do Swagger ou via cURL.

**Acessando a Interface Swagger UI:**
Abra seu navegador e acesse: [http://localhost:8000/docs](http://localhost:8000/docs). Você poderá clicar em "Try it out" na rota `/predict` e enviar uma requisição diretamente pelo navegador.

**Usando cURL (Exemplo de requisição no terminal):**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "Male",
  "age": 67.0,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}'
```

**Exemplo de Resposta Esperada:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_status": "Alto Risco"
}
```

---

## 🐙 Integração Contínua (CI/CD) com GitHub Actions

O projeto possui um workflow configurado (`.github/workflows/docker-publish.yml`) para automatizar a criação e publicação da imagem Docker no **GitHub Container Registry (GHCR)**.

Sempre que um novo push é feito na branch `main` ou uma nova tag de release é gerada, o workflow:
1. Realiza o checkout do código.
2. Faz login no GitHub Container Registry utilizando o token automático do GitHub.
3. Extrai os metadados (tags e labels).
4. Constrói a imagem Docker baseada no `Dockerfile` contendo a API FastAPI e o modelo de ML.
5. Publica a imagem no repositório correspondente (`ghcr.io/joaonovo/fiap_10iadt-tech_challenge_fase_1:latest`).

Dessa forma, a imagem fica disponível para ser baixada e executada em qualquer servidor com Docker instalado de forma facilitada:

```bash
docker pull ghcr.io/joaonovo/fiap_10iadt-tech_challenge_fase_1:latest
docker run -d -p 8000:8000 ghcr.io/joaonovo/fiap_10iadt-tech_challenge_fase_1:latest
```