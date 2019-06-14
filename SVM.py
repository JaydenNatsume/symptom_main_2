# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from pandas import DataFrame


def load_data():
    # Load test data and drop the first column.
    data_form_key_pre = pd.read_csv("DataTitle.csv", header=None, encoding="gbk")
    # Change it's type to array.
    data_form_key_array = np.array(data_form_key_pre)
    # Change it's type to list.
    data_form_key_list = data_form_key_array.tolist()
    data_form_key = []
    for i in data_form_key_list:
        data_form_key = i
    # Get the list of  data form's value
    data_form_value = list(data_form_key_pre)
    # Get the dictionary of test data.
    data_form = dict(zip(data_form_key, data_form_value))
    return data_form


def input_data():
    # Initial a list with all the data is 1.
    test_data = list(np.zeros(208))
    # Input and split symptom.
    print("请输入症状：（不同症状用逗号隔开。例如，“发热，咳嗽，咽喉痛”）")
    input_symptom = input().split('，')
    # Get data form dictionary.
    data_form = load_data()
    # Generate the test data.
    for symptom in input_symptom:
        if symptom not in data_form:
            print(symptom, "不是规范输入！")
        else:
            test_data[data_form[symptom]] = 1
    return test_data, input_symptom


def calculate():
    data = pd.read_csv("Data.csv", header=None)
    y = data[0]
    x = data.drop(0, axis=1).astype(float)
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    # Use DecisionTreeClassifier to predict result.
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(x_train, y_train)
    # Accuracy
    accuracy = clf.score(x_test, y_test)
    print("当前数据准确度为：" + str(accuracy))
    # Predict the input data.
    predict_num, input_symptom = input_data()
    predict = clf.predict([predict_num])
    return predict, input_symptom


def output_stander(predict_num):
    predict = []
    for pre in predict_num:
        if pre == 0:
            predict.append("热")
        elif pre == 1:
            predict.append("寒")
        elif pre == 2:
            predict.append("虚")
        elif pre == 3:
            predict.append("病位属性")
        elif pre == 4:
            predict.append("实")
        elif pre == 5:
            predict.append("气滞")
        elif pre == 6:
            predict.append("水饮")
        elif pre == 7:
            predict.append("气逆")
        elif pre == 8:
            predict.append("湿热")
        elif pre == 9:
            predict.append("湿")
        elif pre == 10:
            predict.append("气虚")
        elif pre == 11:
            predict.append("少阳")
        elif pre == 12:
            predict.append("里")
        elif pre == 13:
            predict.append("表")
        elif pre == 14:
            predict.append("血瘀")
        elif pre == 15:
            predict.append("津伤")
        elif pre == 16:
            predict.append("血虚")
        elif pre == 17:
            predict.append("痰湿")
        elif pre == 18:
            predict.append("营卫不和")
        elif pre == 19:
            predict.append("里热")
        elif pre == 20:
            predict.append("心阳虚")
    return predict[0]


def predict_syndrome(input_symptom, syndrome_element):
    syndrome_data = pd.read_csv("syndrome_syndrome_element_symptom.csv", encoding='gbk')
    syndrome_df = DataFrame(columns=["syndrome_name", "syndrome_element", "main_complain", "main_symptom"])
    syndrome = []
    main_symptom = []
    for i, sy_data in syndrome_data.iterrows():
        if syndrome_element in sy_data["syndrome_element"]:
            syndrome_df.loc[i] = [sy_data["syndrome_name"], sy_data["syndrome_element"],
                                  sy_data["main_complain"], sy_data["main_symptom"]]
    for sy_input in input_symptom:
        for j, syndrome_data in syndrome_df.iterrows():
            if sy_input in syndrome_data['main_complain']:
                syndrome.append(syndrome_data['syndrome_name'])
                main_symptom.append(syndrome_data['main_symptom'])
    print("预测证型为：", syndrome)
    print("预测主症为：", main_symptom)


def start():
    predict, input_symptom = calculate()
    syndrome_element = output_stander(predict)
    print("预测症候属性为：", syndrome_element)
    predict_syndrome(input_symptom, syndrome_element)


start()
