from flask import Flask, render_template, send_file
import pandas as pd
# import prediction function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_csv')
def generate_csv():
    # data = (run prediction function) 
    df = pd.DataFrame(data)
    filename = 'predictions.csv'
    df.to_csv(filename, index=False)
    return send_file(
        filename,
        as_attachment=True,
        download_name='predictions.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)