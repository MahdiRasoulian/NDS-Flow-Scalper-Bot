* سپس، محیط مجازی را مجدداً ایجاد کنید:
# تنظیمات واحدها (XAUUSD):
# - POINT_SIZE = 0.01 (هر point برابر 0.01 دلار)
# - PIP_SIZE = 0.1 (هر pip برابر 10 پوینت)
# - SPREAD_MAX_PIPS بر حسب pip است؛ یعنی مقدار واقعی اسپرد مجاز = SPREAD_MAX_PIPS * PIP_SIZE
# ایجاد محیط مجازی
python -m venv venv



cd "D:\Mahdi\New Backend\V-3\nds_bot"
.\venv\Scripts\activate
set FLASK_APP=main.py

dir /s /b *.py


python main.py


pip install -r requirements.txt
pip install ta_lib-0.6.8-cp313-cp313-win_amd64.whl
