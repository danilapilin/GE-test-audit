import pandas as pd
from numpy import array, arange, maximum, sqrt
import benford as bf
import matplotlib.pyplot as plt
import holidays


def read_files(file_name: str = 'JE input.xlsx', sheet_name: str = 'JE') -> tuple:
    """
    Reading files
    :param file_name: excel file name
    :param sheet_name: name of sheet
    """
    excel_data_df = pd.read_excel(file_name, sheet_name=sheet_name).fillna(0)
    excel_data_df['date1'] = pd.to_datetime(excel_data_df['date'])
    out_of_balance_notes = pd.read_excel('notes.xlsx', sheet_name='Out of balance notes')
    credit_no_expence_notes = pd.read_excel('notes.xlsx', sheet_name='Credit no expence notes', dtype={'Acc Dt': str})

    return excel_data_df, out_of_balance_notes, credit_no_expence_notes


def plot_digs(df, x, y_Exp, y_Found, N, figsize, conf_Z, name_pic, text_x=False) -> None:
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


def handle_datasets(excel_data_df: pd.DataFrame, out_of_balance_notes: pd.DataFrame) -> tuple:
    # by debit (count and amount)
    df1 = excel_data_df.groupby(['Acc Dt']).agg({'Amount': ['count', 'sum']})
    # by credit (count and amount)
    df2 = excel_data_df.groupby(['Acc Ct']).agg({'Amount': ['count', 'sum']})
    # by date
    df3 = excel_data_df.groupby(['Unnamed: 0']).agg({'Amount': ['count', 'sum']})

    out_of_balance_dt_acc = pd.merge(excel_data_df, out_of_balance_notes, on='Acc Dt', how='inner')
    # 4. Top 10 Numerous entries
    top_10_entries = excel_data_df.groupby(['Acc Dt']).agg({'Amount': ['count', 'sum']}).head(10)

    return df1, df2, df3, out_of_balance_dt_acc, top_10_entries


def belford_test(excel_data_df: pd.DataFrame) -> tuple:
    excel_data_df_for_test = excel_data_df.loc[excel_data_df.Amount >= 10]

    # добавляем столбцы для выбора нужных проводок теста Белфорда
    excel_data_df_for_test['Amount'] = pd.to_numeric(excel_data_df_for_test['Amount'])
    excel_data_df_for_test['First_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[0:2]))
    excel_data_df_for_test['First_3_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[0:3]))
    excel_data_df_for_test['Last_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[-2:]))

    digs = 2
    f2d = bf.first_digits(excel_data_df_for_test.Amount, digs=2, decimals=0, confidence=99.99999, high_Z='pos')
    f2d_graf = f2d.sort_values('First_2_Dig')

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

    return excel_data_df_for_test, inner_join_df2


def last_3_digits(excel_data_df_for_test: pd.DataFrame) -> pd.DataFrame:
    digs = 2
    l2d = bf.last_two_digits(excel_data_df_for_test.Amount, decimals=2, confidence=99.99999, high_Z='pos')
    l2d.reset_index(level=0, inplace=True)
    l2d = l2d.loc[l2d.Last_2_Dig > 9]
    # Выбираем числа, которые выше доверительного интервала
    Last_2_Dig_result = l2d.loc[l2d.Found > l2d.Expected].head()
    l2d = l2d.sort_values('Last_2_Dig')
    # сохраняем график
    plot_digs(l2d, x=arange(10, 100), y_Exp=l2d.Expected, y_Found=l2d.Found, N=99.99999, conf_Z=0.00001,
              figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
              name_pic="Last_two_digits.pdf")

    inner_join_l2d = pd.merge(excel_data_df_for_test, Last_2_Dig_result, on='Last_2_Dig', how='inner')
    df = inner_join_l2d.Amount.value_counts().rename_axis('Amount').reset_index(name='counts1')

    inner_join_l2d = pd.merge(inner_join_l2d, df, on='Amount', how='left')
    inner_join_l2d['percent'] = (inner_join_l2d['counts1'] / inner_join_l2d['Counts']) * 100

    return inner_join_l2d


def holidays_ds(excel_data_df: pd.DataFrame) -> pd.DataFrame:
    holidays_day = pd.DataFrame()
    holidays1 = []
    for date in holidays.Russia(years=[2019, 2020]).items():
        holidays1.append(str(date[0]))
    holidays_day['date1'] = holidays1
    holidays_day['date1'] = pd.to_datetime(holidays_day['date1'])
    return pd.merge(excel_data_df, holidays_day, on='date1', how='inner')


def credits_no_expence_rand_choice(excel_data_df, credit_no_expence_notes):
    return pd.merge(excel_data_df, credit_no_expence_notes, on='Acc Dt', how='inner'), \
           excel_data_df.sample(25, random_state=2)


def main() -> None:
    """
    Entry point
    """
    excel_data_df, out_of_balance_notes, credit_no_expence_notes = read_files()
    df1, df2, df3, out_of_balance_dt_acc, top_10_entries = handle_datasets(excel_data_df, out_of_balance_notes)
    excel_data_df_for_test, inner_join_df2 = belford_test(excel_data_df)
    inner_join_l2d = last_3_digits(excel_data_df_for_test)
    inner_join_holidays = holidays_ds(excel_data_df)
    inner_join_credit_no_expence, random_25_rows = credits_no_expence_rand_choice(excel_data_df,
                                                                                  credit_no_expence_notes)
    path = r"JE output.xlsx"
    writer = pd.ExcelWriter(path, engine='openpyxl')
    # excel_data_df.to_excel(writer, 'JE')
    df2.to_excel(writer, 'Acc Dt')
    df1.to_excel(writer, 'Acc Ct')
    df3.to_excel(writer, 'By date')
    out_of_balance_dt_acc.to_excel(writer, 'out_of_balance_acc')
    top_10_entries.to_excel(writer, 'Top_10_entries')
    inner_join_df2.to_excel(writer, 'Benford_law_2_first_digit')
    inner_join_l2d.to_excel(writer, 'Benford_law_2_last_digit')
    inner_join_holidays.to_excel(writer, 'holidays')
    inner_join_credit_no_expence.to_excel(writer, 'Credit_no_expence')
    random_25_rows.to_excel(writer, 'random_choice')
    writer.save()


if __name__ == '__main__':
    main()
