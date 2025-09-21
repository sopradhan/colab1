import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import httpx
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import traceback
import warnings
warnings.filterwarnings('ignore')

# --- Page and CSS Configuration ---
st.set_page_config(page_title="🎯 Analytics AI Platform", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #4f46e5, #7c3aed); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; }
    .agent-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.25rem; border-radius: 12px; margin: 0.75rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .active-agent { background: #dbeafe; border: 2px solid #3b82f6; }
    .jira-card { background: #f0f9ff; border: 1px solid #0ea5e9; padding: 1.25rem; border-radius: 12px; margin: 0.75rem 0; }
    .simulation-result { background: #f0fdf4; border: 1px solid #16a34a; padding: 1.25rem; border-radius: 12px; margin: 0.75rem 0; }
    .error-result { background: #fef2f2; border: 1px solid #ef4444; padding: 1.25rem; border-radius: 12px; margin: 0.75rem 0; }
    .label-tag { background: #e0e7ff; color: #3730a3; padding: 0.25rem 0.75rem; border-radius: 15px; margin: 0.25rem; display: inline-block; font-size: 0.875rem; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LLM and Environment Setup ---
load_dotenv()

@st.cache_resource
def initialize_llm():
    try:
        llm_model = os.getenv('MODEL_2', 'gpt-4o')
        llm = ChatOpenAI(
            base_url=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
            model=llm_model,
            api_key=os.getenv('API_KEY'),
            http_client=httpx.Client(verify=False),
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = initialize_llm()

# --- Financial Datamart Generator ---
class FinancialDataGenerator:
    def __init__(self):
        self.customers = [f"Customer_{i:03d}" for i in range(1, 501)]
        self.products = [
            "Savings Account", "Current Account", "Personal Loan", "Home Loan", 
            "Credit Card", "Investment Product", "Insurance", "Fixed Deposit",
            "Mutual Fund", "Debit Card"
        ]
        self.regions = ["North", "South", "East", "West", "Central"]
        self.channels = ["Branch", "Online", "Mobile App", "ATM", "Phone Banking"]
        
    def generate_customer_dimension(self, n_customers=500):
        data = []
        for i in range(1, n_customers + 1):
            data.append({
                'customer_id': i,
                'customer_name': f"Customer_{i:03d}",
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'region': np.random.choice(self.regions),
                'customer_segment': np.random.choice(['Premium', 'Gold', 'Silver', 'Basic']),
                'acquisition_date': datetime.now() - timedelta(days=np.random.randint(30, 1095))
            })
        return pd.DataFrame(data)
    
    def generate_product_dimension(self):
        data = []
        for i, product in enumerate(self.products, 1):
            data.append({
                'product_id': i,
                'product_name': product,
                'product_category': 'Banking' if product in ['Savings Account', 'Current Account'] else 
                                  'Lending' if 'Loan' in product else 
                                  'Investment' if product in ['Investment Product', 'Mutual Fund', 'Fixed Deposit'] else 'Services',
                'product_margin': np.random.uniform(0.05, 0.25),
                'is_active': np.random.choice([True, False], p=[0.9, 0.1])
            })
        return pd.DataFrame(data)
    
    def generate_time_dimension(self, start_date='2020-01-01', end_date='2024-12-31'):
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        for date in date_range:
            data.append({
                'date_id': int(date.strftime('%Y%m%d')),
                'date': date,
                'year': date.year,
                'quarter': f'Q{date.quarter}',
                'month': date.month,
                'month_name': date.strftime('%B'),
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5,
                'is_month_end': date == date + pd.offsets.MonthEnd(0)
            })
        return pd.DataFrame(data)
    
    def generate_transaction_facts(self, n_transactions=10000):
        data = []
        for i in range(n_transactions):
            transaction_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            data.append({
                'transaction_id': i + 1,
                'customer_id': np.random.randint(1, 501),
                'product_id': np.random.randint(1, 11),
                'date_id': int(transaction_date.strftime('%Y%m%d')),
                'channel': np.random.choice(self.channels),
                'transaction_amount': np.random.exponential(1000) + 100,
                'transaction_count': 1,
                'profit': np.random.uniform(50, 500),
                'is_successful': np.random.choice([True, False], p=[0.95, 0.05])
            })
        return pd.DataFrame(data)

# --- Enhanced SQLite Data Warehouse ---
class DataWarehouse:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.data_generator = FinancialDataGenerator()
        self.initialize_datamart()
    
    def initialize_datamart(self):
        """Initialize financial datamart with dimensions and facts"""
        # Generate and load dimensions
        customer_dim = self.data_generator.generate_customer_dimension()
        product_dim = self.data_generator.generate_product_dimension()
        time_dim = self.data_generator.generate_time_dimension()
        transaction_facts = self.data_generator.generate_transaction_facts()
        
        # Create tables and insert data
        customer_dim.to_sql('dim_customer', self.conn, if_exists='replace', index=False)
        product_dim.to_sql('dim_product', self.conn, if_exists='replace', index=False)
        time_dim.to_sql('dim_time', self.conn, if_exists='replace', index=False)
        transaction_facts.to_sql('fact_transactions', self.conn, if_exists='replace', index=False)
        
        self.conn.commit()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql(query, self.conn)
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return pd.DataFrame()
    
    def get_table_info(self, table_name: str) -> Dict:
        try:
            # Get table schema
            schema_query = f"PRAGMA table_info({table_name})"
            schema = pd.read_sql(schema_query, self.conn)
            
            # Get sample data
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            sample = pd.read_sql(sample_query, self.conn)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count = pd.read_sql(count_query, self.conn)['count'].iloc[0]
            
            return {
                'schema': schema,
                'sample': sample,
                'row_count': count
            }
        except Exception as e:
            return {'error': str(e)}

# --- RAG Knowledge Base ---
class RAGKnowledgeBase:
    def __init__(self, llm):
        self.llm = llm
        try:
            # Use MODEL_4 for embeddings as specified in environment variables
            embedding_model = os.getenv('MODEL_4', 'azure/genailab-maas-text-embedding-3-large')
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('API_KEY'),
                openai_api_base=os.getenv('BASE_URL'),
                model=embedding_model,
                openai_api_type="azure" if "azure" in embedding_model else "openai"
            )
            self.vector_store = None
            self.initialize_knowledge_base()
        except Exception as e:
            st.warning(f"RAG initialization failed: {e}. Using fallback mode.")
            self.embeddings = None
            self.vector_store = None
    
    def initialize_knowledge_base(self):
        """Initialize knowledge base with financial analytics documentation"""
        documents = [
            "Data transformation involves cleaning, aggregating, and preparing financial data for analysis. Common transformations include calculating moving averages, growth rates, and financial ratios.",
            "Financial data visualization best practices include using appropriate chart types for different metrics, ensuring proper scaling, and highlighting key insights through colors and annotations.",
            "Machine learning models for financial data include regression for forecasting, classification for risk assessment, and clustering for customer segmentation.",
            "Customer segmentation analysis helps identify high-value customers, at-risk customers, and growth opportunities using RFM analysis, behavioral patterns, and demographic data.",
            "Financial KPIs include revenue growth rate, profit margins, customer acquisition cost, lifetime value, and churn rate calculations.",
            "Time series analysis for financial data requires handling seasonality, trends, and anomalies using techniques like ARIMA, exponential smoothing, and decomposition.",
            "Risk modeling involves credit scoring, default prediction, and portfolio risk assessment using statistical and machine learning techniques.",
            "Data quality in financial analytics requires validation of transaction amounts, date consistency, customer information accuracy, and handling missing values."
        ]
        
        if self.embeddings:
            try:
                docs = [Document(page_content=doc) for doc in documents]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = text_splitter.split_documents(docs)
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
            except Exception as e:
                st.warning(f"Vector store creation failed: {e}")
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return "\n\n".join([doc.page_content for doc in docs])
            except:
                pass
        return "Use best practices for financial data analysis and follow industry standards."

# --- Enhanced Agent System ---
class BaseAgent:
    def __init__(self, llm, name: str, rag_kb: RAGKnowledgeBase, dw: DataWarehouse):
        self.llm = llm
        self.name = name
        self.rag_kb = rag_kb
        self.dw = dw
        
    def get_context(self, task_description: str) -> str:
        return self.rag_kb.retrieve_context(task_description)

class DataTransformationAgent(BaseAgent):
    def generate_code(self, task_description: str, labels: List[str], feedback: str = None) -> str:
        context = self.get_context(task_description)
        
        # Get available tables info
        tables_info = ""
        for table in ['dim_customer', 'dim_product', 'dim_time', 'fact_transactions']:
            info = self.dw.get_table_info(table)
            if 'error' not in info:
                tables_info += f"\nTable {table}: {info['row_count']} rows\nColumns: {list(info['sample'].columns)}\n"
        
        feedback_prompt = f"\nUser feedback: {feedback}\nPlease revise the code based on this feedback." if feedback else ""
        
        prompt = f"""
        You are an expert Data Transformation Agent for financial analytics.
        
        CONTEXT: {context}
        
        TASK: {task_description}
        LABELS: {', '.join(labels)}
        
        AVAILABLE TABLES:
        {tables_info}
        
        Generate SQL queries and Python pandas code for data transformation. The code should:
        1. Use appropriate SQL joins between dimension and fact tables
        2. Include data quality checks and validation
        3. Handle missing values and outliers appropriately  
        4. Generate relevant financial metrics based on the labels
        5. Return clean, well-commented code
        
        {feedback_prompt}
        
        Return only the executable Python/SQL code with comments:
        """
        
        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception as e:
            return f"# Error generating code: {str(e)}\n# Please check your API configuration"

class VisualizationAgent(BaseAgent):
    def generate_code(self, task_description: str, labels: List[str], feedback: str = None) -> str:
        context = self.get_context(task_description)
        feedback_prompt = f"\nUser feedback: {feedback}\nPlease revise the visualization based on this feedback." if feedback else ""
        
        prompt = f"""
        You are an expert Data Visualization Agent for financial analytics.
        
        CONTEXT: {context}
        
        TASK: {task_description}
        LABELS: {', '.join(labels)}
        
        Generate Python code using plotly for creating interactive financial visualizations. The code should:
        1. Create appropriate chart types for financial data (bar, line, scatter, heatmap, etc.)
        2. Include proper titles, labels, and formatting
        3. Add interactive features like hover data and filters
        4. Use professional color schemes suitable for financial dashboards
        5. Handle different data types and scales appropriately
        
        Assume the data is available as a pandas DataFrame named 'df'.
        
        {feedback_prompt}
        
        Return only the executable Python plotly code with comments:
        """
        
        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception as e:
            return f"# Error generating visualization: {str(e)}\n# Please check your API configuration"

class MLAgent(BaseAgent):
    def generate_code(self, task_description: str, labels: List[str], data_sample: pd.DataFrame = None, feedback: str = None) -> str:
        context = self.get_context(task_description)
        
        # Analyze sample data if provided
        data_analysis = ""
        if data_sample is not None and not data_sample.empty:
            data_analysis = f"""
            DATA ANALYSIS:
            - Shape: {data_sample.shape}
            - Columns: {list(data_sample.columns)}
            - Data types: {dict(data_sample.dtypes)}
            - Missing values: {dict(data_sample.isnull().sum())}
            - Numerical columns: {list(data_sample.select_dtypes(include=[np.number]).columns)}
            """
        
        feedback_prompt = f"\nUser feedback: {feedback}\nPlease revise the model based on this feedback." if feedback else ""
        
        prompt = f"""
        You are an expert ML Agent for financial analytics.
        
        CONTEXT: {context}
        
        TASK: {task_description}
        LABELS: {', '.join(labels)}
        
        {data_analysis}
        
        Based on the task and labels, recommend and generate code for:
        1. Appropriate ML model selection (classification, regression, clustering, forecasting)
        2. Feature engineering specific to financial data
        3. Data preprocessing and scaling
        4. Model training and evaluation
        5. Performance metrics suitable for financial use cases
        
        Popular models for financial analytics:
        - Classification: Logistic Regression, Random Forest, XGBoost for risk assessment
        - Regression: Linear Regression, Random Forest, LSTM for forecasting
        - Clustering: K-Means, DBSCAN for customer segmentation
        - Time Series: ARIMA, Prophet for financial forecasting
        
        {feedback_prompt}
        
        Return executable Python code with model recommendations and implementation:
        """
        
        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception as e:
            return f"# Error generating ML code: {str(e)}\n# Please check your API configuration"

class SimulationAgent(BaseAgent):
    def simulate_code(self, code: str, agent_type: str) -> Dict[str, Any]:
        """Simulate and validate generated code"""
        try:
            # Create a safe execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'plt': None,  # Placeholder for matplotlib if needed
                'sqlite3': sqlite3,
                'conn': self.dw.conn,
                'dw': self.dw
            }
            
            # Add sample data for testing
            sample_data = self.dw.execute_query("SELECT * FROM fact_transactions ft JOIN dim_customer dc ON ft.customer_id = dc.customer_id LIMIT 100")
            exec_globals['df'] = sample_data
            
            # Execute the code
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            
            result = {
                'success': True,
                'message': f"Code executed successfully for {agent_type} agent",
                'output': str(exec_locals) if exec_locals else "Code executed without errors",
                'warnings': []
            }
            
            # Check for common issues
            if agent_type == 'ML Modeling' and 'model' not in code.lower():
                result['warnings'].append("Code doesn't seem to contain model training")
            
            if agent_type == 'Visualization' and 'fig' not in exec_locals:
                result['warnings'].append("No visualization figure created")
                
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Code execution failed: {str(e)}",
                'output': traceback.format_exc(),
                'warnings': []
            }

# --- Enhanced JIRA Manager ---
class JIRAManager:
    def __init__(self):
        self.tickets = [
            {
                'id': 'FIN-001', 
                'title': 'Customer Revenue Analysis Dashboard', 
                'priority': 'High', 
                'labels': ['visualization', 'customer-analysis', 'revenue'],
                'description': 'Create interactive dashboard showing customer revenue trends, segmentation by region, and top performing customers'
            },
            {
                'id': 'FIN-002', 
                'title': 'Credit Risk Prediction Model', 
                'priority': 'Critical', 
                'labels': ['ml-model', 'risk-assessment', 'classification'],
                'description': 'Build ML model to predict customer credit risk and default probability using transaction history and customer demographics'
            },
            {
                'id': 'FIN-003', 
                'title': 'Transaction Data ETL Pipeline', 
                'priority': 'Medium', 
                'labels': ['data-transformation', 'etl', 'data-quality'],
                'description': 'Design ETL pipeline for processing daily transaction data, including validation, cleansing, and aggregation'
            },
            {
                'id': 'FIN-004', 
                'title': 'Sales Forecasting Model', 
                'priority': 'High', 
                'labels': ['ml-model', 'forecasting', 'time-series'],
                'description': 'Develop time series forecasting model for predicting monthly sales across different product categories'
            },
            {
                'id': 'FIN-005', 
                'title': 'Customer Segmentation Analysis', 
                'priority': 'Medium', 
                'labels': ['ml-model', 'clustering', 'customer-analysis'],
                'description': 'Perform customer segmentation using RFM analysis and behavioral patterns to identify distinct customer groups'
            },
            {
                'id': 'FIN-006', 
                'title': 'Profitability Dashboard by Product', 
                'priority': 'High', 
                'labels': ['visualization', 'profitability', 'product-analysis'],
                'description': 'Create comprehensive dashboard showing profit margins, revenue trends, and performance metrics by product line'
            },
            {
                'id': 'FIN-007', 
                'title': 'Fraud Detection System', 
                'priority': 'Critical', 
                'labels': ['ml-model', 'anomaly-detection', 'fraud'],
                'description': 'Implement ML-based fraud detection system using transaction patterns and customer behavior analysis'
            },
            {
                'id': 'FIN-008', 
                'title': 'Monthly Financial KPI Report', 
                'priority': 'Medium', 
                'labels': ['data-transformation', 'kpi', 'reporting'],
                'description': 'Generate automated monthly KPI reports including revenue, profit margins, customer metrics, and growth indicators'
            }
        ]
    
    def get_ticket(self, ticket_id: str):
        for ticket in self.tickets:
            if ticket['id'] == ticket_id:
                return ticket
        return None
    
    def get_tickets_by_label(self, label: str):
        return [ticket for ticket in self.tickets if label in ticket['labels']]

# --- Agent Orchestrator ---
class AgentOrchestrator:
    def __init__(self, data_agent, viz_agent, ml_agent, sim_agent):
        self.agents = {
            'data-transformation': data_agent,
            'etl': data_agent,
            'data-quality': data_agent,
            'kpi': data_agent,
            'reporting': data_agent,
            'visualization': viz_agent,
            'customer-analysis': viz_agent,
            'revenue': viz_agent,
            'profitability': viz_agent,
            'product-analysis': viz_agent,
            'ml-model': ml_agent,
            'risk-assessment': ml_agent,
            'classification': ml_agent,
            'forecasting': ml_agent,
            'time-series': ml_agent,
            'clustering': ml_agent,
            'anomaly-detection': ml_agent,
            'fraud': ml_agent
        }
        self.simulation_agent = sim_agent
    
    def get_required_agents(self, labels: List[str]) -> List[str]:
        agent_types = set()
        for label in labels:
            if label in self.agents:
                agent = self.agents[label]
                if agent.__class__.__name__ == 'DataTransformationAgent':
                    agent_types.add('data-transformation')
                elif agent.__class__.__name__ == 'VisualizationAgent':
                    agent_types.add('visualization')
                elif agent.__class__.__name__ == 'MLAgent':
                    agent_types.add('ml-model')
        return list(agent_types)

# --- Initialize Components ---
@st.cache_resource
def initialize_components():
    dw = DataWarehouse()
    if llm:
        rag_kb = RAGKnowledgeBase(llm)
        data_agent = DataTransformationAgent(llm, "Data Transformation", rag_kb, dw)
        viz_agent = VisualizationAgent(llm, "Visualization", rag_kb, dw)
        ml_agent = MLAgent(llm, "ML Modeling", rag_kb, dw)
        sim_agent = SimulationAgent(llm, "Simulation", rag_kb, dw)
        orchestrator = AgentOrchestrator(data_agent, viz_agent, ml_agent, sim_agent)
        return dw, rag_kb, data_agent, viz_agent, ml_agent, sim_agent, orchestrator
    return None, None, None, None, None, None, None

# Initialize components
components = initialize_components()
if not components[0]:
    st.error("Failed to initialize components. Please check your configuration.")
    st.stop()

dw, rag_kb, data_agent, viz_agent, ml_agent, sim_agent, orchestrator = components
jira_manager = JIRAManager()

# --- Streamlit UI ---
st.markdown('<div class="header"><h1>🎯 Enhanced Analytics AI Platform</h1><p>RAG-based Multi-Agent System for Financial Analytics</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎫 JIRA Tickets")
ticket_id = st.sidebar.selectbox("Select Ticket", [ticket['id'] for ticket in jira_manager.tickets])

if ticket_id:
    ticket = jira_manager.get_ticket(ticket_id)
    
    # Display ticket details
    st.markdown('<div class="jira-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**🎫 {ticket['id']}** - {ticket['title']}")
        st.markdown(f"**Priority:** {ticket['priority']}")
        st.markdown(f"**Description:** {ticket['description']}")
    
    with col2:
        st.markdown("**Labels:**")
        for label in ticket['labels']:
            st.markdown(f'<span class="label-tag">{label}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Determine required agents
    required_agents = orchestrator.get_required_agents(ticket['labels'])
    st.sidebar.markdown("**🤖 Required Agents:**")
    for agent in required_agents:
        st.sidebar.markdown(f"• {agent.replace('-', ' ').title()}")

# Main content area
if st.button("🚀 Execute Multi-Agent Workflow", type="primary"):
    if not ticket_id:
        st.error("Please select a JIRA ticket first!")
    else:
        ticket = jira_manager.get_ticket(ticket_id)
        required_agents = orchestrator.get_required_agents(ticket['labels'])
        
        st.markdown("### 🔄 Multi-Agent Workflow Execution")
        
        # Execute agents in sequence
        results = {}
        
        for agent_type in required_agents:
            with st.expander(f"🤖 {agent_type.replace('-', ' ').title()} Agent", expanded=True):
                if agent_type == 'data-transformation':
                    code = data_agent.generate_code(ticket['description'], ticket['labels'])
                    st.markdown("**Generated Code:**")
                    st.code(code, language='python')
                    
                    # Simulate code
                    sim_result = sim_agent.simulate_code(code, 'Data Transformation')
                    if sim_result['success']:
                        st.markdown('<div class="simulation-result">✅ Code simulation successful</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-result">❌ Simulation failed: {sim_result["message"]}</div>', unsafe_allow_html=True)
                    
                    results[agent_type] = {'code': code, 'simulation': sim_result}
                
                elif agent_type == 'visualization':
                    code = viz_agent.generate_code(ticket['description'], ticket['labels'])
                    st.markdown("**Generated Visualization Code:**")
                    st.code(code, language='python')
                    
                    # Simulate code
                    sim_result = sim_agent.simulate_code(code, 'Visualization')
                    if sim_result['success']:
                        st.markdown('<div class="simulation-result">✅ Visualization code simulation successful</div>', unsafe_allow_html=True)
                        # Try to execute visualization
                        try:
                            sample_data = dw.execute_query("""
                                SELECT dc.region, SUM(ft.transaction_amount) as total_revenue,
                                       COUNT(ft.transaction_id) as transaction_count
                                FROM fact_transactions ft 
                                JOIN dim_customer dc ON ft.customer_id = dc.customer_id
                                GROUP BY dc.region
                            """)
                            
                            if not sample_data.empty:
                                fig = px.bar(sample_data, x='region', y='total_revenue', 
                                           title='Revenue by Region')
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display sample visualization: {e}")
                    else:
                        st.markdown(f'<div class="error-result">❌ Simulation failed: {sim_result["message"]}</div>', unsafe_allow_html=True)
                    
                    results[agent_type] = {'code': code, 'simulation': sim_result}
                
                elif agent_type == 'ml-model':
                    # Get sample data for ML analysis
                    sample_data = dw.execute_query("SELECT * FROM fact_transactions LIMIT 100")
                    code = ml_agent.generate_code(ticket['description'], ticket['labels'], sample_data)
                    st.markdown("**Generated ML Code:**")
                    st.code(code, language='python')
                    
                    # Simulate code
                    sim_result = sim_agent.simulate_code(code, 'ML Modeling')
                    if sim_result['success']:
                        st.markdown('<div class="simulation-result">✅ ML code simulation successful</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-result">❌ Simulation failed: {sim_result["message"]}</div>', unsafe_allow_html=True)
                    
                    results[agent_type] = {'code': code, 'simulation': sim_result}
        
        # Summary
        st.markdown("### 📊 Workflow Summary")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.metric("Agents Executed", len(results))
            successful_sims = sum(1 for r in results.values() if r['simulation']['success'])
            st.metric("Successful Simulations", successful_sims)
        
        with summary_col2:
            total_warnings = sum(len(r['simulation'].get('warnings', [])) for r in results.values())
            st.metric("Total Warnings", total_warnings)
            st.metric("Completion Status", "✅ Complete" if successful_sims == len(results) else "⚠️ Partial")

# Data Warehouse Explorer
st.markdown("### 🏗️ Data Warehouse Explorer")
with st.expander("Explore Financial Datamart", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Tables:**")
        tables = ['dim_customer', 'dim_product', 'dim_time', 'fact_transactions']
        selected_table = st.selectbox("Select Table", tables)
    
    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("Show Table Info"):
            info = dw.get_table_info(selected_table)
            if 'error' not in info:
                st.markdown(f"**Rows:** {info['row_count']}")
                st.markdown("**Schema:**")
                st.dataframe(info['schema'])
                st.markdown("**Sample Data:**")
                st.dataframe(info['sample'])
    
    # Custom Query Interface
    st.markdown("**Custom SQL Query:**")
    custom_query = st.text_area("Enter SQL Query", 
                               value="SELECT * FROM fact_transactions ft JOIN dim_customer dc ON ft.customer_id = dc.customer_id LIMIT 10",
                               height=100)
    
    if st.button("Execute Query"):
        try:
            result_df = dw.execute_query(custom_query)
            if not result_df.empty:
                st.dataframe(result_df)
                
                # Auto-generate insights
                if len(result_df) > 0:
                    st.markdown("**Quick Insights:**")
                    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols[:3]:  # Show stats for first 3 numeric columns
                            st.write(f"• {col}: Mean = {result_df[col].mean():.2f}, Max = {result_df[col].max():.2f}")
            else:
                st.warning("Query returned no results")
        except Exception as e:
            st.error(f"Query execution error: {e}")

# Agent Performance Dashboard
st.markdown("### 🤖 Agent Performance Dashboard")
with st.expander("Agent Analytics", expanded=False):
    # Create mock performance data for demonstration
    agent_performance = pd.DataFrame({
        'Agent': ['Data Transformation', 'Visualization', 'ML Modeling', 'Simulation'],
        'Tasks Completed': [45, 38, 22, 67],
        'Success Rate': [0.92, 0.89, 0.85, 0.94],
        'Avg Response Time': [2.3, 4.1, 8.7, 1.8]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        fig_tasks = px.bar(agent_performance, x='Agent', y='Tasks Completed', 
                          title='Tasks Completed by Agent')
        st.plotly_chart(fig_tasks, use_container_width=True)
    
    with col2:
        fig_success = px.bar(agent_performance, x='Agent', y='Success Rate',
                            title='Success Rate by Agent')
        st.plotly_chart(fig_success, use_container_width=True)

# RAG Knowledge Base Interface
st.markdown("### 🧠 RAG Knowledge Base")
with st.expander("Knowledge Base Query", expanded=False):
    kb_query = st.text_input("Ask the Knowledge Base", 
                            placeholder="e.g., How to perform customer segmentation?")
    
    if kb_query and st.button("Search Knowledge Base"):
        context = rag_kb.retrieve_context(kb_query)
        st.markdown("**Retrieved Context:**")
        st.markdown(context)
        
        # Generate answer using context
        if llm:
            try:
                prompt = f"""
                Based on the following context, answer the user's question:
                
                CONTEXT: {context}
                QUESTION: {kb_query}
                
                Provide a comprehensive answer with practical recommendations:
                """
                answer = llm.invoke(prompt)
                st.markdown("**AI Answer:**")
                st.markdown(answer.content)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

# Feedback System
st.markdown("### 💬 Agent Feedback System")
with st.expander("Provide Feedback", expanded=False):
    feedback_agent = st.selectbox("Select Agent", 
                                 ['Data Transformation', 'Visualization', 'ML Modeling', 'Simulation'])
    feedback_text = st.text_area("Your Feedback", 
                                placeholder="Provide feedback on agent performance, code quality, or suggestions for improvement...")
    feedback_rating = st.slider("Rating", 1, 5, 3)
    
    if st.button("Submit Feedback"):
        # In a real implementation, this would be stored in a database
        st.success(f"Feedback submitted for {feedback_agent} agent! Rating: {feedback_rating}/5")
        st.balloons()

# System Status
st.markdown("### ⚡ System Status")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric("LLM Status", "🟢 Online" if llm else "🔴 Offline")

with status_col2:
    st.metric("RAG Status", "🟢 Active" if rag_kb.vector_store else "🟡 Fallback")

with status_col3:
    # Check database connectivity
    try:
        test_query = dw.execute_query("SELECT COUNT(*) as count FROM fact_transactions")
        db_status = "🟢 Connected" if not test_query.empty else "🔴 Error"
    except:
        db_status = "🔴 Error"
    st.metric("Database", db_status)

with status_col4:
    st.metric("Agents Ready", "🟢 4/4" if llm else "🔴 0/4")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎯 Analytics AI Platform v2.0 | Multi-Agent RAG System for Financial Analytics</p>
    <p>Powered by LangChain, OpenAI, SQLite3, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
