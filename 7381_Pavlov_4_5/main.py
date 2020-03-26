from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logical_exp(a, b, c):
    return ((a != b) and (b != c))

def create_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def element_wise_operations(data, weights):
    activation_func = [relu for _ in range(len(weights)-1)]
    activation_func.append(sigmoid)
    data = data.copy()
    for d in range(len(weights)):
        res = np.zeros((data.shape[0], weights[d][0].shape[1]))
        for i in range(data.shape[0]):
            for j in range(weights[d][0].shape[1]):
                sum = 0
                for k in range(data.shape[1]):
                    sum += data[i][k] * weights[d][0][k][j]
                res[i][j] = activation_func[d](sum + weights[d][1][j])
        data = res
    return data

def numpy_operations(data, weights):
    activation_func = [relu for i in range(len(weights)-1)]
    activation_func.append(sigmoid)
    data = data.copy()
    for i in range(0, len(weights)):
        data = activation_func[i](np.dot(data, weights[i][0]) + weights[i][1])
    return data

def print_predicts(model, data, correct_answer):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_wise_operations(data, weights)
    numpy_res = numpy_operations(data, weights)
    model_res = model.predict(data)
    assert np.isclose(element_wise_res, model_res).all()
    assert np.isclose(numpy_res, model_res).all()
    print("Correct:\n", correct_answer)
    print("Element-wise operations predict:\n", element_wise_res)
    print("Numpy operations predict:\n", numpy_res)
    print("Model predict:\n", model_res)

def main():
    train_data = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 1, 1],
                           [1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 1, 1]])
    correct_answer = np.array([int(logical_exp(*x)) for x in train_data])
    model = create_model()
    print("BEFORE FITTING")
    print_predicts(model, train_data, correct_answer)
    print("AFTER FITTING")
    model.fit(train_data, correct_answer, epochs=200, batch_size=1)
    print_predicts(model, train_data, correct_answer)


if __name__ == '__main__':
        main()