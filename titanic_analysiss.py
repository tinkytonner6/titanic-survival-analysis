import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')

print(df.shape)
print(df.head())
print(df.columns)
print(df.isnull().sum())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())

# Chart 1 - Survival Count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.savefig('survival_count.png')
plt.show()

# Chart 2 - Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.legend(['Died', 'Survived'])
plt.savefig('survival_by_gender.png')
plt.show()

# Chart 3 - Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.legend(['Died', 'Survived'])
plt.savefig('survival_by_class.png')
plt.show()

# Chart 4 - Survival by Age
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Age', hue='Survived', bins=30)
plt.title('Survival by Age')
plt.legend(['Died', 'Survived'])
plt.savefig('survival_by_age.png')
plt.show()

# Chart 5 - Survival by Embarkation Port
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival by Embarkation Port')
plt.legend(['Died', 'Survived'])
plt.savefig('survival_by_embarked.png')
plt.show()
print(df.groupby('Embarked')['Survived'].mean() * 100)

# Create FamilySize column
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Chart 6 - Family Size vs Survival
sns.countplot(x='FamilySize', hue='Survived', data=df)
plt.title('Family Size vs Survival')
plt.legend(['Died', 'Survived'])
plt.savefig('survival_by_familysize.png')
plt.show()

# Chart 7 - Fare Distribution by Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.savefig('fare_by_survival.png')
plt.show()




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Convert Sex to numbers (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numbers
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features for model
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked']
X = df[features]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training size:', X_train.shape)
print('Testing size:', X_test.shape)

# Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', round(accuracy * 100, 2), '%')

# Feature Importance
import pandas as pd
feature_importance = pd.Series(model.feature_importances_, index=features)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()