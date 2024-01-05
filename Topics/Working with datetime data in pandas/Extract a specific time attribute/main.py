import pandas as pd

dob = ['1971-01-07', '1971-10-02', '1987-11-10', '1984-10-02', '1984-12-19', '1986-04-29', '1997-05-04', '2000-08-24', '1992-02-04', '1975-03-02']
df = pd.DataFrame({'date': pd.DatetimeIndex(dob),
                   'name': ['Julia', 'Jake', 'Rose', 'Oliver', 'Kat', 'Brandon', 'Isil', 'Le', 'Laura', 'Mark']})

#Your code here
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year
print(df)