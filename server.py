from flask import Flask, render_template, request
import logic
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/submit',methods=['POST'])
def submit():
  print ('I got clicked!')
  x = int(request.form['search_term'])
  test = [[x]]
  output = logic.predict_and_return_data(test)
  #return output
  return output.to_html(header="true", table_id="table")
  #return 'Clicked by %s' % name

if __name__ == '__main__':
  app.run(debug=True)