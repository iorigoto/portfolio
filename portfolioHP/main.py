from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    code_dict = {}
    file_names = ["pyScrapingP131.py", "step60.py","styleTrans.py","wavenet_multi_audio_updated.py"]  # 読み込むPythonファイルの名前

    for name in file_names:
        with open(f"/Users/pero/Desktop/supreme1/_supreme1/portfolioHP/templates/codes/{name}", "r") as f:
            code_dict[name] = f.read()
    print(code_dict)  # これでコンソールに出力されます

    return render_template("index.html", code_dict=code_dict)

if __name__ == "__main__":
    app.run(debug=True)


