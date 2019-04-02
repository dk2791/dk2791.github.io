---
title: "python code for storing Limit Order Book data of SP500 companies every 5 minute"
date: 2019-04-02
tags: []
excerpt:
mathjax: true
classes: wide
---

Following script will download LOB data of 500 companies every 5 minute into "LOB.csv".
The description of columns in book DataFrames are available [here](https://iextrading.com/developer/docs/#getting-started)


```python

import os
import time
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

print('pandas version :', pd.__version__)

book = web.get_iex_book('AAPL')
list(book.keys())

sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp = pd.read_html(sp_url, header=0)[0] # returns a list for each table


starttime = pd.Timestamp('09:30')
finaltime = pd.Timestamp('16:00')
ref = []
cur_time = pd.Timestamp.now()
empty_set = {}
nl='\n'
tab='\t'
i=0
OtherInfo = pd.DataFrame(columns=["symbol","marketPercent","volume","lastSalePrice","lastSaleSize","lastSaleTime","lastUpdated"])
for item in ['LOI','Trades','SystemEvent','TradingStatus','OpHaltStatus','SsrStatus','TradeBreaks']:
    exec(f'{item} = pd.DataFrame({empty_set})')

while (cur_time - starttime >= pd.Timedelta(0)) and (cur_time - finaltime <= pd.Timedelta(0)):
    starttime = pd.Timestamp.now()
    print(f'start time for iteration number {i}: {starttime}')
    cur_time = pd.Timestamp.now()
    cur_time_string = cur_time.strftime("%H_%M")
    for ticker in sp['Symbol']:
        ticker = ticker.replace('.','_') # BRK.B becomes BRK_B
        exec(f'book_{ticker}_{cur_time_string} = web.get_iex_book("{ticker}")') #change RHS
        # Rounding off the time of fetched data to the nearest minute
        rounding_string = f'if book_{ticker}_{cur_time_string}["lastUpdated"] != 0: {nl}{tab}'
        rounding_string += f'book_{ticker}_{cur_time_string}["lastUpdated"] = pd.Timestamp(book_{ticker}_{cur_time_string}["lastUpdated"], unit="ms").round("min") {nl}'
        rounding_string += f'else: {nl}{tab}book_{ticker}_{cur_time_string}["lastUpdated"] = pd.Timestamp(book_{ticker}_{cur_time_string}["lastUpdated"], unit="ms")'
        exec(rounding_string)
        # Let's create LIMIT ORDER BOOK table
        exec(f'orders_{ticker}_{cur_time_string} = pd.concat([pd.DataFrame(book_{ticker}_{cur_time_string}[side]).assign(side=side) for side in ["bids", "asks"]])')
        exec(f'orders_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'orders_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'LOI=pd.concat([LOI,orders_{ticker}_{cur_time_string}])')

        # Let's create Trade table (by default, last 20 trades per minute)
        exec(f'if "trades" not in book_{ticker}_{cur_time_string}.keys(): book_{ticker}_{cur_time_string}["trades"]=[]')
        exec(f'trades_{ticker}_{cur_time_string} = pd.DataFrame(book_{ticker}_{cur_time_string}["trades"])')
        exec(f'trades_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'trades_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'Trades=pd.concat([Trades,orders_{ticker}_{cur_time_string}])')
        #
        # Let's create System Event table
        exec(f'sysE_{ticker}_{cur_time_string} = pd.DataFrame(book_{ticker}_{cur_time_string}["systemEvent"].items()).T')
        exec(f'if book_{ticker}_{cur_time_string}["systemEvent"]=={empty_set}: sysE_{ticker}_{cur_time_string}["#"]=[None]')
        exec(f'sysE_{ticker}_{cur_time_string}.columns = sysE_{ticker}_{cur_time_string}.loc[0]')
        exec(f'sysE_{ticker}_{cur_time_string}.drop([0],inplace=True)')
        exec(f'sysE_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'sysE_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'SystemEvent=pd.concat([SystemEvent,orders_{ticker}_{cur_time_string}])')

        # Let's create Trading Status table
        exec(f'if "tradingStatus" not in book_{ticker}_{cur_time_string}.keys(): book_{ticker}_{cur_time_string}["tradingStatus"]={empty_set}')
        exec(f'tradeStat_{ticker}_{cur_time_string} = pd.DataFrame(book_{ticker}_{cur_time_string}["tradingStatus"].items()).T')
        exec(f'if book_{ticker}_{cur_time_string}["tradingStatus"]=={empty_set}: tradeStat_{ticker}_{cur_time_string}["#"]=[None]')
        exec(f'tradeStat_{ticker}_{cur_time_string}.columns = tradeStat_{ticker}_{cur_time_string}.loc[0]')
        exec(f'tradeStat_{ticker}_{cur_time_string}.drop([0],inplace=True)')
        exec(f'tradeStat_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'tradeStat_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'TradingStatus=pd.concat([TradingStatus,orders_{ticker}_{cur_time_string}])')

        # Let's create Operational Halt Status table
        exec(f'if "opHaltStatus" not in book_{ticker}_{cur_time_string}.keys(): book_{ticker}_{cur_time_string}["opHaltStatus"]={empty_set}')
        exec(f'opHaltStat_{ticker}_{cur_time_string} = pd.DataFrame(book_{ticker}_{cur_time_string}["opHaltStatus"].items()).T')
        exec(f'if book_{ticker}_{cur_time_string}["opHaltStatus"]=={empty_set}: opHaltStat_{ticker}_{cur_time_string}["#"]=[None]')
        exec(f'opHaltStat_{ticker}_{cur_time_string}.columns = opHaltStat_{ticker}_{cur_time_string}.loc[0]')
        exec(f'opHaltStat_{ticker}_{cur_time_string}.drop([0],inplace=True)')
        exec(f'opHaltStat_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'opHaltStat_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'OpHaltStatus=pd.concat([OpHaltStatus,orders_{ticker}_{cur_time_string}])')

        # Let's create Short Sell Restriction Status table (Shortsell allowed only for uptick)
        exec(f'if "ssrStatus" not in book_{ticker}_{cur_time_string}.keys(): book_{ticker}_{cur_time_string}["ssrStatus"]={empty_set}')
        exec(f'ssrStatus_{ticker}_{cur_time_string} = pd.DataFrame(book_{ticker}_{cur_time_string}["ssrStatus"].items()).T')
        exec(f'if book_{ticker}_{cur_time_string}["ssrStatus"]=={empty_set}: ssrStatus_{ticker}_{cur_time_string}["#"]=[None]')
        exec(f'ssrStatus_{ticker}_{cur_time_string}.columns = ssrStatus_{ticker}_{cur_time_string}.loc[0]')
        exec(f'ssrStatus_{ticker}_{cur_time_string}.drop([0],inplace=True)')
        exec(f'ssrStatus_{ticker}_{cur_time_string}["lastUpdated"] = book_{ticker}_{cur_time_string}["lastUpdated"]')
        exec(f'ssrStatus_{ticker}_{cur_time_string}["symbol"] = book_{ticker}_{cur_time_string}["symbol"]')
        exec(f'SsrStatus=pd.concat([SsrStatus,orders_{ticker}_{cur_time_string}])')

        # Create Other Info table
        exec(f'book = book_{ticker}_{cur_time_string}.copy()')
        exec(f'del book["ssrStatus"]')
        exec(f'del book["tradeBreaks"]')
        exec(f'del book["opHaltStatus"]')
        exec(f'del book["tradingStatus"]')
        exec(f'del book["systemEvent"]')
        exec(f'del book["trades"]')
        exec(f'if "securityEvent" in book.keys():{nl}{tab}del book["securityEvent"]')
        exec(f'del book["bids"]{nl}del book["asks"]')
        exec(f'OtherInfo = OtherInfo.append(book,ignore_index=True)')
    with open(f'./LOI_data/LOI_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as f:
        LOI.to_csv(f, header=f.tell()==0,mode = 'a')
    with open(f'./LOI_data/Trades_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as g:
        Trades.to_csv(g, header=g.tell()==0)
    with open(f'./LOI_data/OtherInfo_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as u:
        OtherInfo.to_csv(u, header=u.tell()==0)
    with open(f'./LOI_data/SystemEvent_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as h:
        SystemEvent.to_csv(h, header=h.tell()==0)
    with open(f'./LOI_data/OpHaltStatus_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as k:
        OpHaltStatus.to_csv(k, header=k.tell()==0)
    with open(f'./LOI_data/SsrStatus_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as l:
        SsrStatus.to_csv(l, header=l.tell()==0)
    with open(f'./LOI_data/TradingStatus_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', 'a') as j:
        TradingStatus.to_csv(j, header=j.tell()==0)

    ref += [cur_time_string]
    endtime = pd.Timestamp.now()
    waittime = starttime + pd.Timedelta('5 minute') - endtime
    time.sleep(waittime.seconds + waittime.microseconds*10**-6)
    endtime2=pd.Timestamp.now()
    print(f'end time for iteration number {i}: {endtime2}')
    print(f'time diff for iteration number {i}: {endtime2-starttime}')
    i+=1

f.close()
g.close()
h.close()
j.close()
k.close()
l.close()
u.close()
```
