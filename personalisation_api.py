from data_calculation import enrich_posts
from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True   # Returns a nicer format and gives good debug reasons - would be switched off for prod

## parameter based call
@app.route('/', methods=['GET'])
def params():
    uid = request.args.get('uid')
    date = request.args.get('date')

    table = enrich_posts(uid,date)
    table = table.to_dict()

    return table

# url based call
@app.route('/<string:uid>/<string:date>', methods=['GET'])
def url(uid:str,date:str):
    table = enrich_posts(uid,date)
    table = table.to_dict()

    return table

if __name__ == '__main__':
    app.run()

