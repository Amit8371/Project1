from flask import Flask, request, Response
from joblib import load

# # Loading my model
my_lr_model = load('new_model.joblib')


# Initialising
app = Flask(__name__)


@app.route("/get_iris_predictions", methods = ['POST','GET'])
def get_iris_predictions():
    # Reading user data, it will look like this {"mydata": '1'}
    data = request.json
    # Getting just his data -> '1'
    user_sent_this_data = data.get('mydata')

    # First converting the user string to a number  -> 1
    user_number = float(user_sent_this_data)

    # You want to use the users data and give it to your model
    model_prediction = my_lr_model.predict([[user_number]])

    # Getting just the number from the array
    my_prediction = model_prediction[0]

    # Giving the user the answer we are looking for
    return Response(str(my_prediction))


if __name__ == '__main__':
    app.run(debug=True)
