import pandas as pd
import benford as bf
import matplotlib.pyplot as plt
from matplotlib import rc
import holidays
from numpy import array, arange, maximum, sqrt
from openpyxl import load_workbook

import os


rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rcParams['pdf.fonttype'] = 42

colors = {'m': '#00798c', 'b': '#E2DCD8', 's': '#9c3848',
          'af': '#edae49', 'ab': '#33658a', 'h': '#d1495b',
          'h2': '#f64740', 't': '#16DB93'}

confs = {None: None, 80: 1.285, 85: 1.435, 90: 1.645, 95: 1.96,
         99: 2.576, 99.9: 3.29, 99.99: 3.89, 99.999: 4.417,
         99.9999: 4.892, 99.99999: 5.327}

dir = os.path.dirname(os.path.abspath(__file__))


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

    plt.savefig('output\\' + name_pic)
    plt.show()


def handle_datasets(excel_data_df: pd.DataFrame, out_of_balance_notes: pd.DataFrame) -> tuple:
    df1 = excel_data_df.groupby(['Acc Dr']).agg({'Amount': ['count', 'sum']})
    df1.columns = ["_".join(x) for x in df1.columns.ravel()]

    df2 = excel_data_df.groupby(['Acc Cr']).agg({'Amount': ['count', 'sum']})
    df2.columns = ["_".join(x) for x in df2.columns.ravel()]

    excel_data_df['date1'] = pd.to_datetime(excel_data_df['date']).dt.date

    df3 = excel_data_df.groupby(['date1']).agg({'Amount': ['count', 'sum']})
    df3.columns = ["_".join(x) for x in df3.columns.ravel()]

    entries_with_no_description = excel_data_df.loc[excel_data_df['Contents / Descryption'].str.len().isna()]

    out_of_balance_dt = out_of_balance_notes['Acc Dr'].dropna()
    out_of_balance_ct = out_of_balance_notes['Acc Cr'].dropna()
    out_of_balance_dt = pd.merge(excel_data_df, out_of_balance_dt, on='Acc Dr', how='inner')
    out_of_balance_ct = pd.merge(excel_data_df, out_of_balance_ct, on='Acc Cr', how='inner')

    return df1, df2, df3, out_of_balance_dt, out_of_balance_ct, entries_with_no_description


def belford_test(excel_data_df: pd.DataFrame) -> tuple:
    excel_data_df_for_test = excel_data_df.loc[excel_data_df.Amount >= 10]

    excel_data_df['Amount_round'] = excel_data_df['Amount'].round()

    excel_data_df_for_test['First_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[0:2]))
    excel_data_df_for_test['Last_2_Dig'] = pd.to_numeric(excel_data_df_for_test['Amount'].apply(lambda x: str(x)[-2:]))

    digs = 2
    f2d = bf.first_digits(excel_data_df_for_test.Amount, digs=2, decimals=0, confidence=99.99999, high_Z='pos')
    f2d_graf = f2d.sort_values('First_2_Dig')

    plot_digs(f2d_graf, x=arange(10, 100), y_Exp=f2d_graf.Expected, y_Found=f2d_graf.Found, N=99.99999, conf_Z=0.00001,
              figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), name_pic="First_two_digits.pdf")

    first_2_dig_result = f2d.loc[f2d.Found > f2d.Expected].head()
    first_2_dig_result.reset_index(level=0, inplace=True)

    inner_join_df1 = pd.merge(excel_data_df_for_test, first_2_dig_result, on='First_2_Dig', how='inner')
    df = inner_join_df1.Amount.value_counts().rename_axis('Amount').reset_index(name='counts1')

    inner_join_df2 = pd.merge(inner_join_df1, df, on='Amount', how='left')
    inner_join_df2['percent'] = (inner_join_df2['counts1'] / inner_join_df2['Counts']) * 100

    inner_join_df2 = inner_join_df2.loc[inner_join_df2.percent > 5]
    inner_join_df2.drop(['Last_2_Dig', 'Found', 'Expected', 'Z_score', 'counts1'], axis=1, inplace=True)

    return excel_data_df_for_test, inner_join_df2, first_2_dig_result


def last_3_digits(excel_data_df_for_test: pd.DataFrame) -> tuple:
    digs = 2

    l2d = bf.last_two_digits(excel_data_df_for_test.Amount, decimals=2, confidence=99.99999, high_Z='pos')
    l2d.reset_index(level=0, inplace=True)
    l2d = l2d.loc[l2d.Last_2_Dig > 9]

    last_2_dig_result = l2d.loc[l2d.Found > l2d.Expected].head()
    l2d = l2d.sort_values('Last_2_Dig')

    plot_digs(l2d, x=arange(10, 100), y_Exp=l2d.Expected, y_Found=l2d.Found, N=99.99999, conf_Z=0.00001,
              figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), name_pic="Last_two_digits.pdf")

    inner_join_l2d = pd.merge(excel_data_df_for_test, last_2_dig_result, on='Last_2_Dig', how='inner')
    df = inner_join_l2d.Amount.value_counts().rename_axis('Amount').reset_index(name='counts1')

    inner_join_l2d = pd.merge(inner_join_l2d, df, on='Amount', how='left')
    inner_join_l2d['percent'] = (inner_join_l2d['counts1'] / inner_join_l2d['Counts']) * 100
    inner_join_l2d = inner_join_l2d.loc[inner_join_l2d.percent > 5]
    inner_join_l2d.drop(['First_2_Dig', 'Found', 'Expected', 'Z_score', 'counts1', 'date1'], axis=1, inplace=True)

    return inner_join_l2d, last_2_dig_result


def get_top_10_entries(excel_data_df: pd.DataFrame) -> tuple:
    ##Debit
    top_10_entries_debit = excel_data_df.groupby(['Acc Dr']).agg({'Amount': ['count', 'sum']})
    top_10_entries_debit.columns = ["_".join(x) for x in top_10_entries_debit.columns.ravel()]
    top_10_entries_debit = top_10_entries_debit.sort_values('Amount_sum', ascending=False).head(10)

    ##Credit
    top_10_entries_credit = excel_data_df.groupby(['Acc Cr']).agg({'Amount': ['count', 'sum']})
    top_10_entries_credit.columns = ["_".join(x) for x in top_10_entries_credit.columns.ravel()]
    top_10_entries_credit = top_10_entries_credit.sort_values('Amount_sum', ascending=False).head(10)

    return top_10_entries_debit, top_10_entries_credit


def get_sum_by_user(excel_data_df: pd.DataFrame) -> pd.DataFrame:
    if 'user id' in excel_data_df.columns:
        sum_by_user = excel_data_df.groupby(['user_id']).agg({'Amount': ['count', 'sum']})
        sum_by_user.columns = ["_".join(x) for x in sum_by_user.columns.ravel()]
        sum_by_user = sum_by_user.sort_values('Amount_sum', ascending=False)
    else:
        sum_by_user = pd.DataFrame(columns=['user_id', 'Amount_count', 'Amount_sum'])

    return sum_by_user


def holidays_ds(excel_data_df: pd.DataFrame) -> tuple:
    holidays_day = pd.DataFrame()
    holidays1 = []
    for date in holidays.Russia(years=[2019, 2020]).items():
        holidays1.append(str(date[0]))
    holidays_day['date1'] = holidays1
    holidays_day['date1'] = pd.to_datetime(holidays_day['date1'])
    excel_data_df['date1'] = pd.to_datetime(excel_data_df['date1'])

    inner_join_holidays = pd.merge(excel_data_df, holidays_day, on='date1', how='inner')

    inner_join_holidays_count = inner_join_holidays.groupby(['date1']).agg({'Amount': ['count', 'sum']})
    inner_join_holidays_count.columns = ["_".join(x) for x in inner_join_holidays_count.columns.ravel()]

    return inner_join_holidays, inner_join_holidays_count


def get_credit_no_expence_notes(excel_data_df: pd.DataFrame, credit_no_expence_notes: pd.DataFrame) -> pd.DataFrame:
    credit_no_expence_notes['Acc Dr'] = pd.to_numeric(credit_no_expence_notes['Acc Dr'], errors='coerce')

    return pd.merge(excel_data_df, credit_no_expence_notes, on='Acc Dr', how='inner')


def check_entry(excel_data_df: pd.DataFrame, list_to_check: list) -> pd.DataFrame:
    d = []

    for k in excel_data_df.values:
        h = 0
        for i in k:
            for number in list_to_check:
                if str(i).find(number) != -1:
                    h += 1
        if h >= 1:
            d.append(k)
    return pd.DataFrame(d, columns=excel_data_df.columns)


def main() -> None:
    """
    Entry point
    """
    """
    excel_data_df, manual_entries, out_of_balance_notes, credit_no_expence_notes, key_words_notes, holidays_added, \
        related_parties_list = read_files()
    """
    data_file = 'data\\' + [i for i in os.listdir(dir + '\\data') if '.xlsx' in i][0]
    notes_file = 'note\\' + [i for i in os.listdir(dir + '\\note') if '.xlsx' in i][0]

    excel_data_df = pd.read_excel(data_file, sheet_name='JE').fillna(0)
    manual_entries = pd.read_excel(data_file, sheet_name='Manual entries')

    out_of_balance_notes = pd.read_excel(notes_file, sheet_name='Out of balance notes')
    credit_no_expence_notes = pd.read_excel(notes_file, sheet_name='Credit no expence notes', dtype={'Acc Dr': str})

    key_words_notes = pd.read_excel(notes_file, sheet_name='Key words')["key words"].tolist()
    holidays_added = pd.read_excel(notes_file, sheet_name='holidays_added')

    related_parties_list = pd.read_excel(notes_file, sheet_name='Related parties_list')[
        "Related parties_list"].tolist()

    df1, df2, df3, out_of_balance_dt, out_of_balance_ct, entries_with_no_description = \
        handle_datasets(excel_data_df, out_of_balance_notes)

    excel_data_df_for_test, inner_join_df2, first_2_dig_result = belford_test(excel_data_df)
    inner_join_l2d, last_2_dig_result = last_3_digits(excel_data_df_for_test)
    top_10_entries_debit, top_10_entries_credit = get_top_10_entries(excel_data_df)

    sum_by_user = get_sum_by_user(excel_data_df)
    related_parties_df = check_entry(excel_data_df, related_parties_list)
    key_words_df = check_entry(excel_data_df, key_words_notes)

    inner_join_holidays, inner_join_holidays_count = holidays_ds(excel_data_df)
    inner_join_credit_no_expence = get_credit_no_expence_notes(excel_data_df, credit_no_expence_notes)

    random_25_rows = manual_entries.sample(25, random_state=2)

    path = r"output\JE output.xlsx"
    #book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    #writer.book = book

    df2.to_excel(writer, 'By debit account')
    df1.to_excel(writer, 'By credit account')
    df3.to_excel(writer, 'By date')
    out_of_balance_dt.to_excel(writer, '1.1 Out of balance debit', index=False)
    out_of_balance_ct.to_excel(writer, '1.2 Out of balance credit', index=False)
    inner_join_df2.to_excel(writer, '2.1 Benfords 2 first digit', index=False)
    first_2_dig_result.to_excel(writer, '2.1 Benfords 2 first digit stat', index=False)
    inner_join_l2d.to_excel(writer, '2.2 Benfords 2 last digit', index=False)
    last_2_dig_result.to_excel(writer, '2.2 Benfords 2 last digit stat', index=False)
    entries_with_no_description.to_excel(writer, '3. Entries with no description', index=False)

    top_10_entries_debit.to_excel(writer, '4.1 Numerous entries debit')
    top_10_entries_credit.to_excel(writer, '4.2 Numerous entries credit')
    sum_by_user.to_excel(writer, '5. By user ID', index=False)
    inner_join_holidays.to_excel(writer, '6.1 Holidays_entries', index=False)
    inner_join_holidays_count.to_excel(writer, '6.2 Holidays_count')
    inner_join_credit_no_expence.to_excel(writer, '7. Credits to expense', index=False)
    related_parties_df.to_excel(writer, '8. Related parties', index=False)
    key_words_df.to_excel(writer, '11. Key words', index=False)
    random_25_rows.to_excel(writer, '10. Manual entries_25 items', index=False)

    writer.save()
    writer.close()


if __name__ == '__main__':
    main()
