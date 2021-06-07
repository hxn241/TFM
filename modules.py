def num_cat(data,num,cat):
    for i in num:
        data[i] = data[i].astype('int64')
    for i in cat:
        data[i] = data[i].astype('category')
    return data

def cat_num(data):
    for i in data.columns:
        if i == 'Age_of_Vehicle' or i == 'Engine_Capacity_(CC)':
            data[i] = data[i].astype('int64')
        else:
            data[i] = data[i].astype('category')
    return data


  

def calculate_vif(X, thresh=5.0):
    from statsmodels.stats.outliers_influence import variance_inflation_factor  
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]


def vtypes(data):
    cat = data.select_dtypes('category').columns
    num = data.select_dtypes('float64').columns
    return cat,num

def return_dtypes_var(data):
    cat = []
    num = []
    for i in data.columns:
        if data[i].dtype == 'float64':
            num.append(i)
        else:
            cat.append(i)
    return cat,num


def lencoder(df_imbalanced):
    lenc = LabelEncoder()
    df_imbalanced[cat] = df_imbalanced[cat].apply(lenc.fit_transform)
    df_imbalanced['Accident_Severity'] = lenc.fit_transform(df_imbalanced['Accident_Severity'])
    return df_imbalanced


def target_features(data):
    X = data.drop('Accident_Severity', axis=1)
    y = data['Accident_Severity']
    return X,y

def split(X,y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test