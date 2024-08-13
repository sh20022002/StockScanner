from concurrent.futures import ThreadPoolExecutor
import subprocess


def run_app():
    subprocess.run(["streamlit", "run",  "app.py"])
    

def run_main():
    subprocess.run(["python", "main.py"])

if 'page' not in st.session_state:
    st.session_state['page'] = 'app'

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_app)
        executor.submit(run_main)