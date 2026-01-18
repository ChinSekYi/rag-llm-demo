install:
	pip install -r requirements.txt

format:
	black .
	isort .


app:
	streamlit run app_interface/app.py