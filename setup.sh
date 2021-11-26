mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘ef0707’
backgroundColor = ‘#a9f7e0’
secondaryBackgroundColor = ‘#ADAC9F’
textColor= ‘#0e0e0e’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
