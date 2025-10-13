# Setting Up Ollama with Streamlit

1. Install Ollama
   ```bash:disable-run
   brew install ollama
   ```

2. Start Ollama Service 
   ```bash
   brew services start ollama
   ```

3. Start Ollama Server
   ```bash
   ollama serve
   ```

4. Download the Model
   ```bash
   ollama pull deepseek-r1:1.5b
   ```

5. Install Python Dependencies 
   ```bash
   uv pip install streamlit ollama
   ```

6. Save and Run the Streamlit App  
   ```bash
   streamlit run ollama_streamlit.py
   ```