import json
import pickle

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = [0] * len(__data_columns)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return __model.predict([x])[0]


def load_saved_artifacts():
    # print("Loading saved artifacts ..... start")
    global __data_columns
    global __locations
    global __model
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/House Price Prediction Project.pickle", "rb") as f:
        __model = pickle.load(f)
    # print("loading saved artifacts ...... Done")


def get_location_names():
    load_saved_artifacts()
    return __locations


if __name__ == "__main__":
    print(get_location_names())
