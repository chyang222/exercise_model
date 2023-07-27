#%%
'''
 @author : ych
'''
''' install '''
# !pip install pystan
# !conda install -c conda-forge fbprophet
# !pip install bs4
# !pip install finance-datareader

''' import '''
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from matplotlib import pyplot as plt
import FinanceDataReader as fdr
import requests
import json
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import re
from sklearn.metrics import mean_absolute_error

''' functions '''
# FinanceDataReader
def data_reader(cryptocurrency, date):
    cryp = fdr.DataReader(cryptocurrency, date)
    print('\ncryptocurrency\n', cryp)
    cryp_df = cryp[['Volume', 'Change', 'Close']].reset_index()

    return cryp_df

# 대한민국 휴일 데이터 - 공공데이터포털 api 활용 수집
def get_holidays_kor(year, month, key):
    holidays = pd.DataFrame([], columns = ['dateKind', 'dateName', 'isHoliday', 'locdate', 'seq'])

    for idx in year:
        for jdx in month:
            y = str(idx)
            m = jdx
            url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo?solYear={}&solMonth={}&ServiceKey={}&_type=json'.format(y, m, key)
            response = requests.get(url)
            raw = json.loads(response.text)
            if raw['response']['body']['items'] != '':
                signs = []
                signs.extend(raw['response']['body']['items']['item'])
                holiday = pd.DataFrame.from_dict(signs)
                if len(holiday.columns) == 5:
                    holidays = pd.concat([holidays, holiday], axis = 0).reset_index(drop = True)

    return holidays

# create holidays dataframe
def holidays(holidays_df):
    holidays = pd.DataFrame({
        'holiday' : list(holidays_df['dateName']),
        'ds' : pd.to_datetime(list(holidays_df['locdate'].apply(lambda x:'{}-{}-{}'.format(str(x)[:4], str(x)[4:6], str(x)[6:])))),
        'lower_window' : 0,
        'upper_window' : 0,
    })
    return holidays

# print max & min row
def max_min_price(forecast):
    today = date.today()
    forecast_fut = forecast[forecast['ds'] >  date.isoformat(today)]
    ff_ds_max = re.split('[\n| ]', str(forecast_fut[forecast_fut['yhat'] == forecast_fut['yhat'].max()]['ds']))[3]
    ff_yhat_max = int(float(re.split('[\n| ]', str(forecast_fut[forecast_fut['yhat'] == forecast_fut['yhat'].max()]['yhat']))[4]))
    ff_ds_min = re.split('[\n| ]', str(forecast_fut[forecast_fut['yhat'] == forecast_fut['yhat'].min()]['ds']))[3]
    ff_yhat_min = int(float(re.split('[\n| ]', str(forecast_fut[forecast_fut['yhat'] == forecast_fut['yhat'].min()]['yhat']))[4]))
    print('\n===========RESULT===========')
    print('Max : {}{:>12,}'.format(ff_ds_max, ff_yhat_max))
    print('Min : {}{:>12,}'.format(ff_ds_min, ff_yhat_min))
    print('============================')
    

''' main '''
if __name__ == '__main__':
    # Input
    #cryptocurrency = input('Please input cryptocurrency(ex. BTC/KRW, ETH/KRW) : ')
    #cryptocurrency = '065450'
    #삼성전자 주가
    cryptocurrency = 'BTC/USD'
    # path = '/home/datanuri/yjjo/1_python_scripts/3_facebook_prophet/3_cryptocurrency'

    # FinanceDataReader
    cryp_df = data_reader(cryptocurrency, '2017-01-01')
    print('\ncryptocurrency_df\n', cryp_df)

    # rename
    # 시계열 : ds
    # 종속변수 : y
    # 독립변수 : add1, add2, add3, ...
    cryp_df.rename(columns = {'Date' : 'ds', 'Volume' : 'add1', 'Change' : 'add2', 'Close' : 'y'}, inplace = True)
    print('\ncryptocurrency_df_rename\n', cryp_df)

    # split train and test
    month = relativedelta(months = 2)
    week = timedelta(weeks = 1)
    before = str(date.today() - week)
    train = cryp_df[cryp_df['ds'] < before]
    test = pd.DataFrame(cryp_df[cryp_df['ds'] >= before]['ds'].reset_index(drop = True))
    test_y = cryp_df[cryp_df['ds'] >= before]['y'].reset_index(drop = True)

    # holidays data
    # holiday_df = get_holidays_kor(range(2017, 2022), ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'ey9nQIK5nOkl%2FfcsfaRczmd%2FHJ%2FQ44LgjF%2FV%2FBLmPlH2beBs4V0zGlG0OYF7PIYKCPX1TctoZ05pS4RP49SlrQ%3D%3D')
    # holiday_df.to_csv('/home/datanuri/yjjo/2_data/holiday_data.csv', index = False, encoding = 'UTF-8-SIG')
    holidays_kor_df = pd.read_csv('/Users/unixking/Desktop/python/prophet botcoin/holiday_data.csv', encoding = 'UTF-8-SIG')
    print('\nholidays_kor_df\n', holidays_kor_df)

    # create holidays dataframe
    holidays_df = holidays(holidays_kor_df)
    print('\nholidays\n', holidays_df)

    # create model
    model = Prophet(
        # growth
        # linear or logistic
        # growth = 'linear', # default : 'linear'

        # trend
        # 값이 높을수록 trend를 유연하게 감지
        changepoint_prior_scale = 0.8,
        # change point 명시
        # changepoints=['2021-02-08', '2021-05-13'],

        # seasonality
        # 연 계절성
        yearly_seasonality = 10, # default : 10 / 0 ~ 50
        # 주 계절성
        weekly_seasonality = 15, # default : 10 / 0 ~ 50
        # 일 계절성
        # daily_seasonality = 10, / 0 ~ 50
        # 계절성 반영 강도
        seasonality_prior_scale = 0.6, # 0 ~ 1
        # additive or multiplicative
        seasonality_mode = 'multiplicative',

        # holidays
        holidays = holidays_df,
        # holiday 반영 강도
        holidays_prior_scale = 10,  # 0 ~ 50
    ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 10) # add month seasonality
    # fit model
    model.fit(train)

    # predict
    past = model.predict(test)
    pred = pd.DataFrame([])
    pred['yhat'] = past['yhat']
    pred['test_y'] = test_y
    pred.index = test['ds']
    plt.plot(pred)
    # plt.savefig(path + '/matplot/compare_pred_test.png')
    plt.show()
    mae = mean_absolute_error(pred['yhat'], pred['test_y'])
    print('MAE: {:.3f}'.format(mae))

    # model components
    model.plot_components(past)
    # plt.savefig(path + '/matplot/cryp_future_components.png')
    # changpoint
    plt.show()

    fig = model.plot(past)
    a = add_changepoints_to_plot(fig.gca(), model, past)
    # plt.savefig(path + '/matplot/cryp_future_change_point.png')
    plt.show()

    # create model
    model2 = Prophet(
        # growth
        # linear or logistic
        # growth = 'linear', # default : 'linear'

        # trend
        # 값이 높을수록 trend를 유연하게 감지
        changepoint_prior_scale = 0.8,
        # change point 명시
        # changepoints=['2021-02-08', '2021-05-13'],

        # seasonality
        # 연 계절성
        yearly_seasonality = 10, # default : 10
        # 주 계절성
        weekly_seasonality = 15, # default : 10
        # 일 계절성
        # daily_seasonality = 10,
        # 계절성 반영 강도
        seasonality_prior_scale = 0.6,
        # additive or multiplicative
        seasonality_mode = 'multiplicative',

        # holidays
        holidays = holidays_df,
        # holiday 반영 강도
        holidays_prior_scale = 10,
    ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 10) # add month seasonality
    # fit model
    model2.fit(cryp_df)

    future = model2.make_future_dataframe(periods = 100)
    
    # create future dataframe
    print('\nfuture\n', future)
    forecast = model2.predict(future)
    model2.plot(forecast)
    
    # plt.savefig(path + '/matplot/cryp_future.png')
    plt.show()

    # print max & min row
    max_min_price(forecast)
# %%
