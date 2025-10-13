# Setting Up Ollama with Streamlit

1. Install Ollama*
   Install the Ollama package using Homebrew:
   ```bash:disable-run
   brew install ollama
   ```

2. Start Ollama Service 
   Start the Ollama service in the background:
   ```bash
   brew services start ollama
   ```

3. Start Ollama Server
   Ensure the Ollama server is running before pulling any models:
   ```bash
   ollama serve
   ```

4. Download the Model
   Pull the DeepSeek R1 model (1.5B parameters):
   ```bash
   ollama pull deepseek-r1:1.5b
   ```

5. Install Python Dependencies 
   Install the required Python packages for Streamlit and Ollama:
   ```bash
   pip install streamlit ollama
   ```

6. Save and Run the Streamlit App  
   Save your Python code as `ollama_streamlit.py` and run the Streamlit application:
   ```bash
   streamlit run ollama_streamlit.py
   ```