import flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True
@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h2><p>This site is a prototype API for SOÑAR CON UN FUTURO SIN BOOTCAMPS.</p>"
app.run()