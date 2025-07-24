# PCOS Diagnostic Assistant (AI Agent)

## Project Overview:

This project introduces an intelligent AI agent designed to assist individuals in understanding their potential risk of Polycystic Ovary Syndrome (PCOS). It leverages a machine learning model to predict the likelihood of PCOS based on various health parameters provided by the user. Beyond just a prediction, the system acts as a conversational assistant, offering clear explanations for the diagnosis and providing actionable recommendations for next steps. The aim is to empower users with initial insights, encouraging them to seek professional medical advice where indicated.

**Key achievements of this project include:**
* **Building an End-to-End AI Agent:** Orchestrating multiple tools and an LLM to provide a comprehensive diagnostic flow.
* **Integrating Machine Learning for Prediction:** Seamlessly incorporating a pre-trained PCOS prediction model into the agent's workflow.
* **Contextual Explanation and Recommendation:** Generating personalized, human-readable insights and advice based on model predictions.
* **Robust Data Logging:** Implementing a persistent database to record all user interactions and agent outputs for analysis and improvement.

## The Problem I'm Solving

In healthcare, early awareness and understanding of potential conditions are crucial. PCOS is a complex hormonal disorder often characterized by varied symptoms, leading to delayed diagnosis. Individuals may be uncertain about their symptoms or what steps to take.

This project addresses the need for an accessible, informative, and user-friendly tool that can:
* Provide an **initial assessment** of PCOS likelihood based on self-reported data.
* **Demystify the prediction** by explaining which specific health features contribute to the outcome.
* Offer **clear, practical next steps** and lifestyle recommendations.

By bridging the gap between raw data and understandable health insights, this agent helps reduce information overload and empowers users to make more informed decisions about their health journey.

## Core Features That Make It Work

This AI agent employs a sophisticated workflow to deliver predictions, explanations, and recommendations:

* **1. Predictive Machine Learning Model:**
    * **How:** A pre-trained `RandomForestClassifier` model is used to assess the probability of PCOS based on 40+ clinical and lifestyle features. This model was meticulously trained and evaluated to achieve high accuracy.
    * **Benefit:** Provides a data-driven prediction (Likely PCOS / Unlikely PCOS) along with a confidence score.

* **2. Intelligent Agent Orchestration (LangGraph):**
    * **How:** The agent's flow is managed by `langgraph`, enabling a robust, multi-step process. It dynamically calls specialized tools for prediction, explanation, and recommendation based on the current state.
    * **Benefit:** Ensures a structured, reliable, and intelligent interaction from initial data input to final recommendations.

* **3. Specialized Diagnostic Tools:**
    * **`predict_pcos`**: Processes raw patient data, maps it to the model's required features, and executes the ML prediction.
    * **`explain_pcos_results`**: Analyzes the prediction and key contributing features (e.g., irregular cycles, AMH levels) to generate a concise, easy-to-understand explanation.
    * **`recommend_next_steps`**: Offers tailored advice, suggesting professional consultation and lifestyle adjustments for "Likely PCOS," or general health maintenance for "Unlikely PCOS."
    * **Benefit:** Decouples complex logic into manageable, reusable functions that the AI can invoke.

* **4. Interactive Command-Line Interface:**
    * **How:** A Python script provides a simple command-line interface that guides the user through entering their health data. It uses Pydantic for strict data validation.
    * **Benefit:** Makes the system easy to interact with for demonstration and local testing purposes.

* **5. Persistent Data Logging (SQLite):**
    * **How:** Every full diagnostic session, including the user's input, the prediction, explanation, and recommendations, is saved to a SQLite database (`pcos_agent_logs.db`).
    * **Benefit:** Allows for historical tracking of interactions, potential model monitoring, and future analysis.

## 🛠 Technologies Used

* **Python 3.x**
* **AI Agent Framework:** `langgraph`
* **Large Language Model (LLM):** `ChatGroq` (`llama-3.1-8b-instant`) for agent reasoning and tool calling.
* **Machine Learning:** `scikit-learn` (for `RandomForestClassifier`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), `imblearn` (`SMOTETomek` for handling imbalanced data).
* **Data Manipulation:** `pandas`, `numpy`
* **Data Validation:** `pydantic`
* **Model Persistence:** `joblib`
* **Database:** `sqlite3`
* **Environment Management:** `python-dotenv`
* **Other:** `json`, `re`, `uuid`, `os`, `functools` (`lru_cache`).

## 📂Files

* `agent_tools.py`: Defines the core Python tools (`predict_pcos`, `explain_pcos_results`, `recommend_next_steps`) that the AI agent can call.
* `ai_agent.py`: Implements the `langgraph` agent, orchestrating the calls to the tools and managing the overall flow.
* `interactive_agent.py`: Provides the command-line interface for user interaction and feeds data into the AI agent.
* `memory.py`: Handles database operations (saving logs to `pcos_agent_logs.db`).
* `model.ipynb`: Jupyter Notebook detailing the data preprocessing, model training, evaluation, and saving of the PCOS prediction model.
* `graph.ipynb`: Jupyter Notebook for visualizing the `langgraph` agent's workflow.
* `pcos_model_revised.joblib`: The saved pre-trained machine learning model and its required columns.
* `pcos_agent_logs.db`: The SQLite database file containing logs of all agent interactions.
* `PCOS_data.csv`: (Assumed source for `model.ipynb`) The dataset used for training the PCOS prediction model.
* `.env.example`: Example file for environment variables (e.g., `GROQ_API_KEY`, `PCOS_DB_PATH`).

## 🚀 Getting Started

Follow these steps to set up and run the PCOS Diagnostic Assistant on your machine:

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL_HERE] # Replace with your actual GitHub URL
    cd [your-repository-name]
    ```

2.  **Install Required Libraries:**
    It's recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
    Then, install the dependencies. You'll likely need to manually create a `requirements.txt` based on the `🛠 Technologies Used` section, or install them one by one:
    ```bash
    pip install langgraph langchain-groq pydantic python-dotenv joblib pandas scikit-learn imbalanced-learn
    ```
    *(Note: `langchain-core` is a dependency of `langgraph` and `langchain-groq`.)*

3.  **Set Up Environment Variables:**
    * Create a file named `.env` in the root of your project directory.
    * Add your Groq API key (for the LLM) to this file:
        ```
        GROQ_API_KEY="your_groq_api_key_here"
        ```
    * (Optional) If you want to specify a different path for the database, add:
        ```
        PCOS_DB_PATH="path/to/your/database.db"
        ```
    * You can obtain a Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys).

4.  **Ensure Model and Data Files are Present:**
    * Make sure `pcos_model_revised.joblib` is in your project's root directory. This file contains the pre-trained ML model.
    * Ensure `PCOS_data.csv` (the dataset used for training the model) is also present if you plan to rerun `model.ipynb`.

## ▶️ How to Use the PCOS Diagnostic Assistant

1.  **Run the Interactive Agent:**
    * Open your terminal or command prompt in the project folder.
    * Execute the `interactive_agent.py` script:
        ```bash
        python interactive_agent.py
        ```
    * The agent will greet you and prompt you to enter various health and lifestyle metrics. Follow the on-screen instructions, providing numerical inputs (0 for No, 1 for Yes where applicable).

2.  **Review the Output:**
    * After you provide all the required information, the agent will process it, make a prediction, provide an explanation, and offer recommendations directly in your terminal.

3.  **Inspect the Logs (Optional):**
    * You can open `pcos_agent_logs.db` with a SQLite browser to see all recorded interactions.

## Data Source

The machine learning model for this project was trained using a dataset related to PCOS, typically sourced from health surveys or clinical data. The specific dataset used in `model.ipynb` is `PCOS_data.csv`.

##  What I Learned & What Makes This Project Stand Out

This project provided invaluable hands-on experience in:

* **Applied AI Agent Development:** Moving beyond simple chatbots to create a multi-step, tool-using AI agent with `langgraph`.
* **ML Model Integration:** Understanding how to effectively load, integrate, and leverage a custom-trained machine learning model within an LLM-powered application.
* **Tool-Use Engineering:** Designing discrete, well-defined tools (`predict_pcos`, `explain_pcos_results`, `recommend_next_steps`) that enhance the agent's capabilities.
* **Data-Driven Explanations:** Crafting logic to provide transparent explanations for ML predictions, improving user trust and understanding.
* **Robust Input Handling:** Implementing `pydantic` for strict data validation, ensuring the agent receives clean and correctly formatted inputs.
* **Persistence and Logging:** Setting up a basic database for logging, critical for debugging, analysis, and future improvements.

## 📈 Future Enhancements

I envision several ways to expand and improve this diagnostic assistant:

* **Web-Based User Interface:** Develop a front-end (e.g., using React or HTML/CSS/JS) for a more user-friendly and accessible experience.
* **Enhanced Explanations:** Incorporate more nuanced and personalized explanations by leveraging the LLM's generative capabilities more extensively.
* **Long-Term Health Tracking:** Allow users to save their profiles and track changes over time, potentially enabling personalized health trends and alerts.
* **Medical Disclaimer Integration:** Explicitly add and emphasize professional medical disclaimers within the application.
* **Advanced Data Visualization:** Integrate data visualization libraries (e.g., Matplotlib, Seaborn) to visually represent key health metrics in `model.ipynb` or a future UI.
* **Error Handling & User Feedback:** Improve robustness by adding more graceful error handling and clearer feedback for invalid inputs.

## Author

**Taylor Wanyama**  
[LinkedIn Profile](https://www.linkedin.com/in/taylor-wanyama-421920271/)

