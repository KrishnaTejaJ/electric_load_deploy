from flask import Flask, render_template
import pickle
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, validators, FloatField, SelectField
import pandas as pd
import numpy as np
#pd.set_option('display.max_colwidth', -1)
# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'
bootstrap = Bootstrap(app)

# load the model from disk
model = pickle.load(open('model/model.pkl', 'rb'))
feature_names= ['Load (T-24h)', 'Load (T-48h)', 'Load (T-72h)', 'Load (T-168h)', 'Load (T-336h)', 'Load (T-504h)']

class FeaturesForm(FlaskForm):
    load_24hrs = FloatField('L(T-24)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
    load_48hrs = FloatField('L(T-48)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
    load_72hrs = FloatField('L(T-72)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
    load_168hrs = FloatField('L(T-168)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
    load_336hrs = FloatField('L(T-336)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
    load_504hrs = FloatField('L(T-504)', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])

    submit = SubmitField('Submit')
    @app.route('/', methods=['GET','POST'])
    def predict():
        form = FeaturesForm()
        if form.validate_on_submit():
            load_24hrs = form.load_24hrs.data
            load_48hrs = form.load_48hrs.data
            load_72hrs = form.load_72hrs.data
            load_168hrs = form.load_168hrs.data
            load_336hrs = form.load_336hrs.data
            load_504hrs = form.load_504hrs.data

            features = [load_24hrs, load_48hrs, load_72hrs, load_168hrs, load_336hrs, load_504hrs]
            features_val = [load_24hrs, load_48hrs, load_72hrs, load_168hrs, load_336hrs, load_504hrs]
            df = pd.DataFrame([features], columns=feature_names)
            prediction = model.predict([features_val])
            print(prediction)
            result = prediction[0]
            return render_template('result.html', df = df, result=result)
        return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
