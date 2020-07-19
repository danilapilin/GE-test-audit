import pandas as pd
from numpy import array, arange, maximum, sqrt
import benford as bf
import matplotlib.pyplot as plt
import holidays

# 0. Выгрузка журнала проводок и справочников

excel_data_df = pd.read_excel('JE testing (Titan Russia)_12m19.xlsx', sheet_name='JE').fillna(0)
excel_data_df['date1'] = pd.to_datetime(excel_data_df['date'])
Out_of_balance_notes = pd.read_excel('notes.xlsx', sheet_name='Out of balance notes')
Credit_no_expence_notes = pd.read_excel('notes.xlsx', sheet_name='Credit no expence notes')

# 1. Journal entries summarised:

# by debit (count and amount)
df1 = excel_data_df.groupby(['Acc Dt']).agg({'Amount': ['count', 'sum']})
# by credit (count and amount)
df2 = excel_data_df.groupby(['Acc Ct']).agg({'Amount': ['count', 'sum']})
# by date
df3 = excel_data_df.groupby(['Unnamed: 0']).agg({'Amount': ['count', 'sum']})

# 2. Out of balance account
out_of_balance_dt_acc = pd.merge(excel_data_df, Out_of_balance_notes, on='Acc Dt', how='inner')

# 3.Benford's law
# 3.1 проверка 2 первых цифр
excel_data_df_for_test = excel_data_df.loc[excel_data_df.Amount >= 10]

# добавляем столбцы для выбора нужных проводок теста Белфорда
excel_data_df_for_test['Amount'] = pd.to_numeric(excel_data_df_for_test['Amount'])
excel_data_df_for_test['First_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[0:2]))
excel_data_df_for_test['First_3_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[0:3]))
excel_data_df_for_test['Last_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[-2:]))

digs = 2
f2d = bf.first_digits(excel_data_df_for_test.Amount, digs=2, decimals=0, confidence=99.99999, high_Z='pos')
f2d_graf = f2d.sort_values('First_2_Dig')


# Функция, которая рисует график, нужно дорабатывать

def plot_digs(df, x, y_Exp, y_Found, N, figsize, conf_Z, name_pic, text_x=False):
    if len(x) > 10:
        rotation = 90
    else:
        rotation = 0
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Expected vs. Found Distributions', size='xx-large')
    plt.xlabel('Digits', size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    if conf_Z is not None:
        sig = conf_Z * sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower_zeros = array([0] * len(upper))
        lower = maximum(y_Exp - sig - (1 / (2 * N)), lower_zeros)
        u = (y_Found < lower) | (y_Found > upper)
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, zorder=5)
        ax.plot(x, lower, zorder=5)
        ax.fill_between(x, upper, lower,
                        alpha=.3, label='Conf')
    ax.bar(x, y_Found * 100., label='Found', zorder=3, align='center')
    ax.plot(x, y_Exp * 100., linewidth=2.5,
            label='Benford', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    if text_x:
        ind = array(df.index).astype(str)
        ind[:10] = array(['00', '01', '02', '03', '04', '05',
                          '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    ax.set_ylim(0, max([y_Exp.max() * 100, y_Found.max() * 100]) + 10 / len(x))
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.savefig(name_pic)
    plt.show()


# рисуем график и сохраняем в pdf:
plot_digs(f2d_graf, x=arange(10, 100), y_Exp=f2d_graf.Expected, y_Found=f2d_graf.Found, N=99.99999, conf_Z=0.00001,
          figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5))
          , name_pic="First_two_digits.pdf")

# Выбираем числа, которые выше доверительного интервала
First_2_Dig_result = f2d.loc[f2d.Found > f2d.Expected].head()
First_2_Dig_result.reset_index(level=0, inplace=True)

inner_join_df1 = pd.merge(excel_data_df_for_test, First_2_Dig_result, on='First_2_Dig', how='inner')
df = inner_join_df1.Amount.value_counts().rename_axis('Amount').reset_index(name='counts1')

inner_join_df2 = pd.merge(inner_join_df1, df, on='Amount', how='left')
inner_join_df2['percent'] = (inner_join_df2['counts1'] / inner_join_df2['Counts']) * 100

inner_join_df2 = inner_join_df2.loc[inner_join_df2.percent > 5]

# 3.2. проверка 3 первых цифр
l2d = bf.last_two_digits(excel_data_df_for_test.Amount, decimals=2, confidence=99.99999, high_Z='pos')
l2d.reset_index(level=0, inplace=True)
l2d = l2d.loc[l2d.Last_2_Dig > 9]
# Выбираем числа, которые выше доверительного интервала
Last_2_Dig_result = l2d.loc[l2d.Found > l2d.Expected].head()
l2d = l2d.sort_values('Last_2_Dig')
# сохраняем график
plot_digs(l2d, x=arange(10, 100), y_Exp=l2d.Expected, y_Found=l2d.Found, N=99.99999, conf_Z=0.00001,
          figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5))
          , name_pic="Last_two_digits.pdf")

inner_join_l2d = pd.merge(excel_data_df_for_test, Last_2_Dig_result, on='Last_2_Dig', how='inner')
df = inner_join_l2d.Amount.value_counts().rename_axis('Amount').reset_index(name='counts1')

inner_join_l2d = pd.merge(inner_join_l2d, df, on='Amount', how='left')
inner_join_l2d['percent'] = (inner_join_l2d['counts1'] / inner_join_l2d['Counts']) * 100

# 4. Top 10 Numerous entries
Top_10_entries = excel_data_df.groupby(['Acc Dt']).agg({'Amount': ['count', 'sum']}).head(10)

# 5. group by user
# нужно добавить

# 6. Holidays
holidays_day = pd.DataFrame()
holidays1 = []
for date in holidays.Russia(years=[2019, 2020]).items():
    holidays1.append(str(date[0]))
holidays_day['date1'] = holidays1
holidays_day['date1'] = pd.to_datetime(holidays_day['date1'])
inner_join_holidays = pd.merge(excel_data_df, holidays_day, on='date1', how='inner')

# 7. Credits no expense

Credit_no_expence_notes = pd.read_excel('notes.xlsx', sheet_name='Credit no expence notes', dtype={'Acc Dt': str})
inner_join_Credit_no_expence = pd.merge(excel_data_df, Credit_no_expence_notes, on='Acc Dt', how='inner')

# 8.Random choice
# выбираем 25 рандомных строк с повторением
random_25_rows = excel_data_df.sample(25, random_state=2)

# 11. Key words
# В процессе пока не знаю, как проверить по нескольким полям сразу

# Result
# Выгрузка результатов в ексель

path = r"JE testing (Titan Russia)_12m19_new.xlsx"
writer = pd.ExcelWriter(path, engine='openpyxl')
# excel_data_df.to_excel(writer, 'JE')
df2.to_excel(writer, 'Acc Dt')
df1.to_excel(writer, 'Acc Ct')
df3.to_excel(writer, 'By date')
out_of_balance_dt_acc.to_excel(writer, 'out_of_balance_acc')
Top_10_entries.to_excel(writer, 'Top_10_entries')
inner_join_df2.to_excel(writer, 'Benford_law_2_first_digit')
inner_join_l2d.to_excel(writer, 'Benford_law_2_last_digit')
inner_join_holidays.to_excel(writer, 'holidays')
inner_join_Credit_no_expence.to_excel(writer, 'Credit_no_expence')
random_25_rows.to_excel(writer, 'random_choice')
writer.save()
