from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    result1 = "None"
    data = pd.read_csv(r'/Users/kushalraghuwanshi/Desktop/diabetes 3.csv')

    if request.method == 'GET' and 'n1' in request.GET:
        try:
            X = data.drop("Outcome", axis=1)
            Y = data['Outcome']
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

            model = LogisticRegression()
            model.fit(x_train, y_train)

            param_names = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']

            # Empty list to store the values
            values = []

            # Loop through each parameter and convert to float
            for name in param_names:
                param_value = request.GET.get(name)
                if param_value:
                    try:
                        value = float(param_value)
                        values.append(value)
                    except ValueError:
                        values.append(0.0)
                        pass
                else:
                    values.append(0.0)
                    pass

            # You can now access the converted values as follows:
            val1, val2, val3, val4, val5, val6, val7, val8 = values

            pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

            result1 = ""

            if pred == [1]:
                result1 = "Positive"
            else:
                result1 = "Negative"

        except ValueError:
            result1 = "Invalid Input case"

    return render(request, "predict.html", {"result2": result1})

