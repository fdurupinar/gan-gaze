import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
def load_merged_features(window_size):


    fetures_path = "../features/merged_features_"+ str(window_size) + ".csv"
    personality_path = "../features/merged_traits_" + str(window_size) + ".csv"

    f_ws = pd.read_csv(fetures_path, sep=',')
    p_ws = pd.read_csv(personality_path, sep=',')

    df = pd.DataFrame({'mean':f_ws['mean blink duration'], 'var':f_ws['var blink duration'],
                         'min': f_ws['min blink duration'],'max': f_ws['max blink duration'], 'rate': f_ws['blink rate'],
                         'N': p_ws['Neuroticism'], 'E': p_ws['Extraversion'], 'O': p_ws['Openness'], 'A': p_ws['Agreeableness'],'C': p_ws['Conscientiousness']
                          })

    return df

def fit_personality(window_size):

    df = load_merged_features(window_size)
    all_blink_var_names = ['mean', 'var', 'min', 'max', 'rate']

    y = df['mean'].to_numpy()

    X = df.drop(all_blink_var_names + ['E', 'O', 'A', 'C'], axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

    # Scale the data to be between -1 and 1
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # random forest model creation
    rfc = RandomForestRegressor()
    rfc.fit(X_train, y_train)


    # predictions
    rfc_predict = rfc.predict(X_test)

    # print(rfc_predict)
    mse = mean_squared_error(y_test, rfc_predict)
    rmse = np.sqrt(mse)
    print(rmse)

    # print(rfc.score(X_test, y_test))

    # Calculate mean absolute percentage error (MAPE)
    # errors = abs(rfc_predict - y_test)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')



    p3 = [3]
    p2 = [2]
    p1 = [1]
    pred1 = rfc.predict([p1])
    pred2 = rfc.predict([p2])
    pred3 = rfc.predict([p3])

    print(pred1)
    print(pred2)
    print(pred3)

    plt.plot(range(1), pred1[0],'ro')
    plt.plot(range(1), pred2[0],'bo')
    plt.plot(range(1), pred3[0],'go')
    plt.show()

    return rfc
    # arr2d = np.array(['N','E','O','A','C'] + blink_var_names)
    # print NEOAC
    for c in range(1, 4):
        for a in range(1, 4):
            for o in range(1, 4):
                for e in range(1, 4):
                    for n in range(1, 4):
                        p = [n, e, o, a, c]
                        f_ext = '_'.join(map(str, p))

                        pred = rfc.predict([p])

                        with open('out/blink_values_' + f_ext + '.csv', 'w') as fp:
                            np.savetxt(fp, pred, delimiter=',')

                        # new_row = np.concatenate((p, pred[0]), axis=0)
                        # arr2d = np.vstack((arr2d, new_row))


    # np.savetxt('out/blink_values.csv', arr2d, delimiter=',', fmt='%s')

    return rfc

    # print("=== Confusion Matrix ===")
    # print(confusion_matrix(y_test, rfc_predict))
    # print('\n')
    # print("=== Classification Report ===")
    # print(classification_report(y_test, rfc_predict))
    # print('\n')


rfc = fit_personality(5)

