---
title: "Project Number 1"
permalink: /ml/ml-project-1/
header:
  image: "/images/image5.jpeg"
sidebar:
  nav: "ml"
classes: wide
---

# NLP & Regression on SP500 Companies

#### Does a popular NLP model, LDA, applied to earnings transcript provide valuable information beyond those from common financial indicators?

![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im1.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im2.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im3.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im4.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im5.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im6.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im7.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im8.png)
![alt]({{ site.url}}{{ site.baseurl }}/images/ml_project_1/im9.png)


Note: After implementing this notebook, I tried to trim down a little bit. Thus, I have removed some unnecessary cells


```
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pickle
import os
chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver
```

## 1. Scraping data

### 1.1 Scraping company symbol

Let's scrape SP500 company information from wikipedia


```
sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp = pd.read_html(sp_url, header=0)[0]
sp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Security</th>
      <th>Symbol</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3M Company</td>
      <td>MMM</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>St. Paul, Minnesota</td>
      <td>NaN</td>
      <td>66740</td>
      <td>1902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbott Laboratories</td>
      <td>ABT</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AbbVie Inc.</td>
      <td>ABBV</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABIOMED Inc</td>
      <td>ABMD</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Danvers, Massachusetts</td>
      <td>2018-05-31</td>
      <td>815094</td>
      <td>1981</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Accenture plc</td>
      <td>ACN</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Scraping Earnings Transcripts from Nasdaq

Scrape earnings transcript from SP500 companies. Activate # to scrape pages that dynmically load from javascript. However, another problem was encountered; that was, the captcha. Thus, I ignored the dynamic pages


```


my_dict = defaultdict(str)
#BRK.B, CPRI, SCHW have no earnings call transcript for Q3 2018
SYMBOLS = sp['Symbol'].drop([71,90,105])

cnt=0
captcha=[]
captcha2=[]
for symbol in SYMBOLS:
    url_path = f'https://www.nasdaq.com/symbol/{symbol}/call-transcripts'
    response = requests.get(url_path)
    soup = BeautifulSoup(response.text,'lxml')
    query = re.compile('Q3 2018')
    if not soup.find(text=query):
        # If page is dynamically loading from javascript, I couldn't locate href. Then, use selenium
        # to obtain rendered page
        alter_url=f'https://seekingalpha.com/symbol/{symbol}/earnings/transcripts'
        # driver = webdriver.Chrome(chromedriver)
        # driver.get(url_path)
        # soup = BeautifulSoup(driver.page_source,'lxml')
        # soup.find_all('a')
        response = requests.get(alter_url)
        soup = BeautifulSoup(response.text,'lxml')
        if soup.find(text=re.compile('captcha')) != None:
            captcha.append(symbol)
            continue
        aug_path = soup.find(text=query).find_parent()['href']
        call_url = f'https://seekingalpha.com'+aug_path
        call_response = requests.get(call_url)
        soup = BeautifulSoup(call_response.text,'lxml')
        if soup.find(text=re.compile('captcha')) != None:
            captcha2.append(symbol)
            continue
        raw_transcript = [item.text for item in soup.find(text=query).find_parent().find_parent().find_parent().find_all('p')]
        raw_transcript=raw_transcript[raw_transcript.index('')+1:]
        raw_transcript=''.join(raw_transcript)
    else:
        unrefined_string = soup.find(text=query).find_parents('span')
        aug_path = unrefined_string[0].find('a')['href']
        call_url = f'http://www.nasdaq.com'+aug_path

        call_response = requests.get(call_url)
        soup = BeautifulSoup(call_response.text,'lxml')
        raw_transcript = [item.text for item in soup.find(text=query).find_parent().find_parent().find_parent().find_all('p')]
        raw_transcript=''.join(raw_transcript)
        if raw_transcript.find('Operator') != -1:
            raw_transcript=raw_transcript[raw_transcript.find('Operator')+8:]
        else:
            raw_transcript=raw_transcript[raw_transcript.find('Presentation')+5:]

    exec(f'my_dict["{symbol}"] += raw_transcript[:raw_transcript.find("Read the rest of")]')
    cnt+=1
    if cnt%10 ==0: print(f"{cnt} companies' transcript read")

```

    480 companies' transcript read


The list of companies filtered by Captcha were as following


```
print(captcha, captcha2)
```

    ['GOOGL', 'CNP', 'CTAS', 'CSCO', 'ED', 'DISCA', 'DOW', 'EXPD', 'JEF', 'LLL', 'LIN', 'MDT', 'MU', 'MSFT', 'NWS', 'PGR', 'CRM', 'TROW', 'FOX', 'UA', 'WMT'] []


The list of transcripts are as following


```
my_dict
```




    defaultdict(str,
                {'MMM': "I would now like to turn the call over to Bruce Jermeland, Director of Investor Relations at 3M.Bruce JermelandThank you and good morning, everyone. Welcome to our third quarter 2018 business review. With me today are Mike Roman, 3M's Chief Executive Officer; and Nick Gangestad, our Chief Financial Officer. Mike and Nick will make some formal comments, and then we'll take your questions. Please note that today's earnings release and slide presentations accompanying this call are posted on our Investor Relations website at 3m.com under the heading Quarterly Earnings.Before we begin, let me remind you of the dates for our upcoming investor events found on Slide 2. First, we will be hosting our Investor Day at your headquarters in St. Paul, Minnesota in a few weeks with a welcome reception in the evening of Wednesday, November 14th where we will be highlighting how 3M Science is advancing our priority markets for growth. Along with the formal presentation program, on Thursday November 15, the presentations will discuss our new five-year plan along with our preview of our 2019 outlook. If you plan to attend the event and have not yet responded, please RSVP right away. Second, our Q4 Earnings Conference call will take place on Tuesday, January 29, 2019.Please take a moment to read the forward-looking statement on Slide 3. During today's conference call, we will make certain predictive statements that reflect our current views about 3M's future performance and financial results. These statements are based on certain assumptions and expectations of future events that are subject to risks and uncertainties. Item 1A of our most recent Form 10-K lists some of the most important risk factors that could cause actual results to differ from our predictions.Please note that throughout today's presentation, we'll be making references to certain non-GAAP financial measures. Reconciliations of the non-GAAP measures can be found in the appendices of today's presentation and press release. Please turn to Slide 4 and I'll hand out to Mike. Mike?Mike RomanThank you, Bruce. Good morning, everyone and thank you, for joining us. In the third quarter 3M delivered a double-digit increase and cash flow and earnings per share along with strong margins despite slower growth. We also continue to execute on business transformation or deploying capital to invest in our future and return cash to our shareholders. Looking at the numbers, our team posted total sales of $8.2 billion in the quarter.We delivered organic growth of 1% which is on top of 7% growth in last year's third quarter. As you recall from the discussion on our July earnings call, our ERP rollout in the US resulted in revenue shifting between Q2 and the second half of the year. Today, we estimate a pull forward into Q2 was approximately 100 basis points with the vast majority coming out of Q3.Moving on to earnings per share, we posted EPS of $2.58, an increase of 11% year-over-year. Our company continues to deliver strong return on invested capital along with premium margins.Companywide we generated margins of 25% with all business groups above 22%. Our team also increased free cash flow by 24% year-over-year with a conversion rate of 114%. This is a testament to the strength of portfolio and business model, and our focus on driving productivity every day. We also continue to invest in R&D and capital to support organic growth while returning cash to our shareholders. And in the quarter, we returned $1.9 billion to 3M shareholders through both dividends and share repurchases.Please turn to Slide 5 for look at the performance of our business groups for both the third quarter and year-to-date. In the third quarter, three of our five business groups, Electronics and Energy, Industrial and Safety and Graphics posted organic growth of 2%. Healthcare and Consumer each had areas of strength, but also areas that were softer than expected. Healthcare's growth declined by 1% primarily due to continued weakness in our drug delivery business. Organic growth in Consumer was down 2%. This business group was impacted by channel adjustments between quarters with our major retail customers, though the sellout of our products remains strong. In his comments, Nick will provide more detail on third quarter performance of each business group.Given the shift of sales between quarters due to business transformation, it is also helpful to look at our performance through nine months. Safety and Graphics posted 6% growth followed by 3% growth from both Industrial and Electronics and Energy. Healthcare posted 2% growth with Consumer at 1%. Companywide, we have delivered organic growth of more than 3% year-to-date. I will come back to share our updated guidance after Nick takes us through the details of the quarter. Nick?",
                 'ABT': "With the exception of any participant's questions asked during the question-and-answer session, the entire call, including the question-and-answer session, is material copyrighted by Abbott. It cannot be recorded or rebroadcast without Abbott's expressed written permission.And I would now like to introduce Mr. Scott Leinenweber, Vice President, Investor Relations, Licensing and Acquisitions. Scott LeinenweberGood morning and thank you for joining us. With me today are Miles White, Chairman of the Board and Chief Executive Officer; and Brian Yoor, Executive Vice President, Finance and Chief Financial Officer. Miles will provide opening remarks and Brian will discuss our performance and outlook in more detail. Following their comments, Miles, Brian and I will take your questions.Before we get started, some statements made today may be forward-looking for purposes of the Private Securities Litigation Reform Act of 1995, including the expected financial results for 2018. Abbott cautions that these forward-looking statements are subject to risks and uncertainties that may cause actual results to differ materially from those indicated in the forward-looking statements. Economic, competitive, governmental, technological and other factors that may affect Abbott's operations are discussed in Item 1a, Risk Factors to our Annual Report on Securities and Exchange Commission Form 10-K for the year ended December 31, 2017.Abbott undertakes no obligation to release publicly any revisions to forward-looking statements as a result of subsequent events or developments, except as required by law. Please note that third quarter financial results and guidance provided on the call today for sales, EPS and line items of the P&L will be for continuing operations only.On today's conference call, as in the past, non-GAAP financial measures will be used to help investors understand Abbott's ongoing business performance. These non-GAAP financial measures are reconciled with the comparable GAAP financial measures in our earnings news release and regulatory filings from today, which are available on our website at abbott.com.Unless otherwise noted, our commentary on sales growth refers to organic sales growth, which adjusts the 2017 basis of comparison to exclude the impact of exchange and historical results for Abbott's medical optics and St. Jude's vascular closure businesses, which were divested during the first quarter of 2017, as well as current and prior year sales for Alere which was acquired on October 3, 2017.With that, I will now turn the call over to Miles.Miles WhiteOkay. Thanks, Scott, and good morning. Today, we reported results of another strong quarter with ongoing earnings per share of $0.75 along with sales growth of approximately 8% on an organic basis, reflecting well balanced growth across all four of our businesses. I'm particularly pleased with the continued productivity of our new product pipeline and would like to highlight a couple areas where our products are creating and fundamentally shaping markets. I'll start with Structural Heart, where we're the world or the global leader in minimally invasive treatments for mitral regurgitation or a leaky heart valve. We've recently made several significant advancements in this area. In July, in U.S., we initiated a pivotal trial for Tendyne, our device that is designed to replace damaged mitral heart valves without the need for open heart surgery. We also received U.S. FDA approval for our third generation version of MitraClip, our market leading device for the repair of mitral heart valve. And in September, we announced results of our landmark COAPT trial, which demonstrated that MitraClip improves survival and clinical outcomes for patients with functional mitral regurgitation, the most prevalent form of this condition. We expect to submit this study data to the U.S. FDA in the coming weeks to support consideration of an expanded indication for MitraClip. These advancements will further enhance and strengthen our leadership position in this large and highly underpenetrated disease area and will bring new therapies to patients for effective treatment options that are currently limited. Diabetes Care is another area where our technologies are making a big impact, specifically FreeStyle Libre, a revolutionary glucose monitoring system that eliminates the need for routine fingersticks. In the U.S., we received FDA approval for a 14-day sensor with a shorter one-hour warmup making Libre the longest lasting wearable glucose sensor available. And in Europe, we obtained CE Mark for our FreeStyle Libre 2 system, our newest generation 14-day system, with optional real-time alarms. In a relatively short period of time, FreeStyle Libre now has more than 1 million users across the globe, a testament to the mass market appeal of this product, which is fundamentally changing the way people with diabetes manage their disease. I'll now summarize our third quarter results in more detail before turning call over to Brian. I'll start with Diagnostics, where sales grew 7.5% in the quarter. Alinity, our family of highly differentiated instruments is achieving accelerated growth and strong competitive win rates in Europe where more than 50% of our Alinity instrument placements thus far are coming from share capture. The global rollout of Alinity positions this business for a consistent above market growth for years to come as we capture share and bring the full suite of systems to additional geographies including the U.S. ",
                 'ABBV': "I would now like to introduce Ms. Liz Shea, Vice President of Investor Relations.Elizabeth Shea - AbbVie, Inc.Good morning and thanks for joining us. Also on the call with me today are: Rick Gonzalez, Chairman of the Board and Chief Executive Officer; Michael Severino, Executive Vice President of Research and Development and Chief Scientific Officer; Bill Chase, Executive Vice President of Finance and Administration; and Rob Michael, Senior Vice President and Chief Financial Officer.Before we get started, I would like to remind you that statements we make today are or may be considered forward-looking statements for purposes of the Private Securities Litigation Reform Act of 1995. AbbVie cautions that these forward-looking statements are subject to risks and uncertainties that may cause actual results to differ materially from those indicated in the forward-looking statements. Additional information about the factors that may affect AbbVie's operations is included in our 2017 Annual Report on Form 10-K and in our other SEC filings. AbbVie undertakes no obligation to release publicly any revisions to forward-looking statements as a result of subsequent events or developments except as required by law.On today's conference call, as in the past, non-GAAP financial measures will be used to help investors understand AbbVie's ongoing business performance. These non-GAAP financial measures are reconciled with comparable GAAP financial measures in our earnings release and regulatory filings from today which can be found on our website. Following our prepared remarks we'll take your questions.So with that I'll now turn the call over to Rick.Richard A. Gonzalez - AbbVie, Inc.Thank you, Liz. Good morning, everyone, and thank you for joining us today. I'll discuss our third quarter performance, our full-year 2018 guidance, and I'll provide some comments about our expectations for 2019. Mike will then provide an update on recent advancements across our R&D pipeline and Bill will discuss the quarter in more detail. Following our remarks, as usual, we will take your questions.We delivered another quarter of outstanding performance with results once again ahead of our expectations. We've driven strong commercial, operational, and R&D execution resulting in industry-leading top and bottom line growth. Adjusted earnings per share were $2.14, representing growth of more than 50% versus last year. Total adjusted operational sales growth of 18.5% was driven by a number of products in our portfolio including HUMIRA with global operational sales growth of nearly 10% and IMBRUVICA which grew more than 40% versus the prior year. HCV was a substantial contributor in the quarter with more than $860 million in sales, and we also saw strong performance from several other products including VENCLEXTA, CREON, DUODOPA and LUPON. Based on the continued strength of our business in the quarter and our progress year to date, we are raising our 2018 earnings guidance for the fourth time this year. We now expect full-year 2018 adjusted earnings per share of $7.90 to $7.92, reflecting growth of more than 41% at the midpoint. This represents exceptional earnings growth and puts us at the very top of our industry peer group.And while 2018 certainly sets a very high bar for performance, we expect strong earnings growth once again next year. Although it's too early to provide specific growth targets for 2019 as we're still in the midst of our annual planning process, based on our strong underlying business momentum, we are confident in our ability to deliver double digit earnings growth once again in 2019. And importantly, we expect to deliver this level of growth despite a number of factors including direct biosimilar competition, impacting our more than $6 billion international HUMIRA business; a difficult comparison year, particularly in light of the rapid ramp of our HCV business in 2018; and the significant investment we will be making in 2019 to support new product launches including VENCLEXTA, ORILISSA, risankizumab and upadacitinib. Clearly 2019 will be an important year for AbbVie, an opportunity for us to clearly demonstrate our ability to continue to drive strong earnings growth.In addition to the strong bottom line performance, we also expect to continue to generate significant cash flows. Underscoring the continued confidence in our business, today we're announcing 11.5% increase in our quarterly cash dividend from $0.96 per share to $1.07 per share beginning with the dividend payable in February 2019. Since our inception, we have grown our quarterly dividend nearly 170%.Also in the quarter, we announced two additional patent license agreements over proposed HUMIRA biosimilar products. These agreements, of which there are now five in total, are yet another example of the strength of our intellectual property, and we remain confident that we will not see direct biosimilar competition in the U.S. until 2023.As we look at our business and evaluate our prospects, just about every aspect of our business is performing at or above our expectations. Our pipeline remains one of the best in the industry and the progress we're making to bring new products to the market will allow us to support our long-term growth expectations. Let me highlight a few examples. We have invested significantly in oncology over the past several years to build a major new growth driver for AbbVie. Today our hematological oncology portfolio is now annualizing above $4 billion and growing at a robust rate, including growth of more than 48% in the third quarter.",
                 'ABMD': "I would now like to introduce your host for today's conference, Ingrid Goldberg, Director of Investor Relations. Please go ahead.Ingrid GoldbergThank you. Good morning and welcome to Abiomed's third quarter fiscal 2018 earnings conference call. This is Ingrid Goldberg, Director of Investor Relations for Abiomed, and I'm here with Mike Minogue, Abiomed's Chairman, President and Chief Executive Officer, and Bob Bowen, Former Chief Financial Officer and Consultant to Abiomed.The format for today's call will be as follows: First, Mike Minogue will discuss strategic highlights from the third fiscal quarter and then turn to key operational and strategic objectives. Next, Bob Bowen will provide details on the financial results outlined in today's press release. We will then open the call for your questions.Before we begin, I would like to remind you that comments made on today's call may contained forward-looking statements about the Company's progress relating to financial performance, clinical, regulatory, commercial, litigation and other matters. These forward-looking statements are subject to certain risks and uncertainties that may cause actual results to differ materially from those expressed or implied in the forward-looking statements.Additional information regarding these risks and uncertainties are described in the earnings release we issued this morning, our Annual Report on Form 10-K for the year ended March 31, 2017, and our most recently filed Quarterly Report on Form 10-Q. We undertake no obligation to update or revise any of these forward-looking statements as a result of subsequent events or developments, except as required by law.During today's call adjusted net income and adjusted net income per diluted share and non-GAAP measures will be used to help investors understand the Company's financial performance. These financial measures are reconciled with the comparable GAAP financial measures in the earnings release and regulatory filing from today which are available in the Investor Section of our website at www. abiomed.com.Thank you for joining us. I'm now pleased to introduce, Abiomed's Chairman, President and Chief Executive Officer, Mike Minogue.Michael MinogueThanks, Ingrid. Good morning, everyone. In Q3, Abiomed delivered best ever quarterly results on nearly every metric starting with $154 million in revenue, up 34%. Our robust growth set new records and was driven by U.S. patient utilization for both protected PCI and emergent support, which increased 24% and 43% respectively. In Company history, our top three U.S. Impella utilization months were all in Q3.We also achieved records in Europe aided by strength in Germany which grew 71% year-over-year. Our operational discipline and execution allowed us to achieve best-in-class gross margin of 84% and our best ever operating margin of 29%. We generated an additional $57 million in cash from operations, which brings our cash position to $351 million. We now own both manufacturing and training facilities in Massachusetts and Aachen, Germany and maintain no debt as a company.With our solid balance sheet, we are investing in innovation, education, and our patent portfolio. This quarter, we marked a new milestone of 300 patents awarded and an additional 272 patents pending. It is an exciting time as we collaborate with our customers to improve patient outcomes and the standard of care for circulatory support. As a result, the field of heart recovery is growing, and Abiomed is well positioned to capture the $5 billion U.S. market opportunity while planting the seeds for future growth in global markets.For today's call, I will highlight the clinical need and growing awareness of heart recovery with Impella support in the U.S., Germany, and Japan. Starting with the U.S., both the protected PCI and cardiogenic shock indications established new quarterly records. Our Impella adoption increased to a total of 9% of the 121,000 high-risk PCI and 100,000 emergency patients. We continue to have a long runway for growth because of our clinical benefits and FDA approvals.Our focus on training data and time combined with independent physician-led initiatives like CHIP and the National CSI continues to improve awareness and is having an impact on adoption. Last quarter, I spoke about a publication from the Advisory Board, an independent expert healthcare consultant company acquired by United Healthcare. This quarter, the Advisory Board circulated two additional publications referencing Impella.The first publications comes from the Cardiovascular Roundtable Annual Meeting, protected PCI, complex coronary artery disease and cardiogenic shock were highlighted as advanced programs that enable innovative hospitals to differentiate themselves and capture new market opportunities. As a reference TAVR and mitral valve are also on this list.The second publication stressed the significant clinical need for the growing cardiogenic shock population. The report outlined the necessity for aligned patient care recommended tactics for building a cardiogenic shock network and addressed the benefits ranging from improved quality of care to alleviated cost burden. This timely publication piggybacks on the success of this year's TCT physician call to ARMS around improving outcomes in cardiogenic shock and this initiative is underway across United States.",
                 'ACN': 'I would now like to turn the conference over to our host and facilitator as well as our Managing Director, Head of Investor Relations, Angie Park. Please go ahead.Angie ParkThank you, Steve, and thanks, everyone, for joining us today on our third quarter fiscal 2018 earnings announcement. As the operator just mentioned, I’m Angie Park, Managing Director, Head of Investor Relations.With me today are Pierre Nanterme, our Chairman and Chief Executive Officer; and David Rowland, our Chief Financial Officer. We hope you’ve had an opportunity to review the news release we issued a short time ago.Let me quickly outline the agenda for today’s call. Pierre will begin with an overview of our results. David will take you through the financial details, including the income statement and balance sheet for the third quarter. Pierre will then provide a brief update on our market positioning before David provides our business outlook for the fourth quarter and full fiscal year 2018. We will then take your questions before Pierre provides a wrap up at the end of the call.As a reminder, when we discuss revenues during today’s call, we’re talking about revenues before reimbursements or net revenues. Some of the matters we’ll discuss on this call, including our business outlook are forward-looking and as such, are subject to known and unknown risks and uncertainties, including but not limited to those factors set forth in today’s news release and discussed in our Annual Report on Form 10-K and Quarterly Reports on Form 10-Q and other SEC filings. These risks and uncertainties could cause actual results to differ materially from those expressed on this call.During our call today, we will reference certain non-GAAP financial measures which we believe provide useful information for investors. We include reconciliations of non-GAAP financial measures where appropriate to GAAP in our news release or in the Investor Relations section of our website at accenture.com. As always, Accenture assumes no obligation to update the information presented on this conference call.Now, let me turn the call over to Pierre.Pierre NantermeThank you, Angie, and thanks, everyone, for joining us today. Accenture had a truly outstanding third quarter. We delivered excellent results from new bookings and revenues to operating margin, EPS and cash flow, and we gained significant market share once again.The durability of our performance demonstrates the relevance of our growth strategy and our ability to continue delivering strong results and returns for our shareholders, while at the same time investing significantly in new growth opportunities to strengthen our position for the long-term.Here are a few highlights from the quarter. We delivered record new bookings of $11.7 billion. We grew revenues 11% in local currency to $10.3 billion, and our growth continues to be well-balanced across the dimensions of our business. We delivered earnings per share of $1.79 on an adjusted basis, an 18% increase.Operating margin was 15.7%, an expansion of 20 basis points on an adjusted basis. We generated very strong free cash flow of $1.8 billion, and we returned approximately $1.6 billion in cash to shareholders through share repurchases and the payment of our semiannual dividend.So we’re entering the fourth quarter with excellent momentum in our business, and I feel confident that we are very well-positioned to deliver our business outlook for the year.Now, let me hand over to David, who’ll review the numbers in greater detail. David, over to you?David RowlandThank you, Pierre, and thanks to all of you for taking the time to join us on today’s call. As you heard in Pierre’s comments, we’re extremely pleased with our results in the third quarter, which once again, reflect strong momentum across every dimension of our business.Based on the strength of our third quarter results and the strong confidence and visibility we have in our fourth quarter, we will be increasing key elements of our full-year outlook, which I’ll cover in more detail later in our call. Importantly, both our third quarter results and our updated outlook for the full-year reflect very strong execution against all three financial imperatives for driving superior shareholder value, which I covered in some detail at our Investor Analyst Day in April.So before I get into the details of the quarter, let me summarize the major headlines of our third quarter results. Net revenue increased more than $1.4 billion, reflecting growth of a 11% local currency and representing the third consecutive quarter of double-digit growth.The strong top line growth exceeded our expectations and reflected strong and balanced growth across all operating groups and geographic areas with several growing double digits. The growth continues to significantly outpace the market, reflecting both our leadership position in "the New" and the durability of our diverse yet highly focused growth model.',
                 'ATVI': "Good day, everyone, and welcome to the Activision Blizzard Q3 2018 earnings conference call. Today's conference is being recorded.At this time, I would like to turn the conference over to Christopher Hickey, Senior Vice President of Investor Relations. Please go ahead, sir.Christopher Hickey - Activision Blizzard, Inc.I would like to remind everyone that during this call, we will be making statements that are not historical facts. The forward-looking statements in this presentation are based on information available to the company as of the date of this presentation. And while we believe them to be true, they ultimately may prove to be incorrect.A number of factors could cause the company's actual future results and other future circumstances to differ materially from those expressed in any forward-looking statements. These include the risk factors discussed in our SEC filings, including our 2017 Annual Report on Form 10-K, and those on the slides that are showing. The company undertakes no obligation to release publicly any revisions to any forward-looking statements to reflect events or circumstances after today, November 8, 2018.We will present both GAAP and non-GAAP financial measures during this call. Non-GAAP financial measures exclude the impact of expenses related to stock-based compensation, the amortization of intangible assets and expenses related to acquisitions, including legal fees, costs, expenses, and accruals, expenses related to debt financings and refinancings, restructuring charges, the associated tax benefits of these excluded items, and the impact of certain significant discrete tax-related items.These non-GAAP measures are not intended to be considered in isolation from, as a substitute for, or superior to our GAAP results. We encourage investors to consider all measures before making an investment decision. Please refer to our earnings release, which is posted on www.activisionblizzard.com, for a full GAAP to non-GAAP reconciliation and further explanation with respect to our non-GAAP measures.There's also a PowerPoint overview, which you can access with the webcast and which will be posted to the website following the call. In addition, we will be posting a financial overview highlighting both GAAP and non-GAAP results and a one-page summary.And now I'd like to introduce our CEO, Bobby Kotick.Robert A. Kotick - Activision Blizzard, Inc.Thank you all for joining us today.Our results for the third quarter exceeded our prior outlook, as we continue to entertain large audiences, drive deep engagement, and attract significant audience investment across our franchises. Last quarter, on average, 345 million people played our games each month, and our players spent a record 52 minutes per day playing Activision, Blizzard, and King games.Our unique advantage is the ability to create the most compelling interactive and spectator entertainment based on our own franchises combined with our direct digital connection to hundreds of millions of customers in over 190 countries. With these competitive advantages, we continue to connect and engage the world through epic entertainment.Very few companies are able to consistently deliver compelling content to hundreds of millions of customers. Fewer still can provide their audiences with flexible methods of payment for that content. For our hundreds of millions of customers, we now offer content on phones, computers, and video game consoles, and subscription billing, direct digital download billing, virtual item sales, digital advertising, and of course, we still sell our products through tens of thousands of stores around the world.As an example of the breadth of our capabilities, we launched Call of Duty: Black Ops 4 on October 12. Ordinarily, we launch new Call of Duty titles this week in November. But we believe holiday customers, of which there are millions, will benefit from more players in the game earlier.Our engagement to date is better than any Call of Duty content in recent years, and spectator viewing is higher than ever before. As a franchise, Call of Duty has now generated more revenue than the Marvel Cinematic Universe in the box office, and double that of the cumulative box office of Star Wars. We have an exciting future planned for Call of Duty players, including our new Call of Duty professional player opportunities, and lots of exciting new content in 2019 and beyond.We remain focused on the key growth drivers of our business that we believe present meaningful revenue and engagement upside, including live operations, mobile, and investment in new and growing franchise engagement models. We're pleased with our early momentum in areas like our advertising initiatives, which continue to exceed our plans, as revenues grew almost 50% sequentially.",
                 'ADBE': "Mike SaviageGood afternoon, and thank you for joining us today. Joining me on the call are Adobe's President and CEO, Shantanu Narayen; and John Murphy, Executive Vice President and CFO.In our call today, we will discuss Adobe's third quarter fiscal year 2018 financial results. By now, you should have a copy of our earnings press release which crossed the wire approximately one hour ago. We've also posted PDFs of our earnings call prepared remarks and slides, financial targets and an updated investor datasheet on Adobe.com. If you would like a copy of these documents, you can go to Adobe's Investor Relations page and find them listed under Quick Links.Before we get started, we want to emphasize that some of the information discussed in this call, particularly our revenue and operating model targets and our forward-looking product plans, is based on information as of today, September 13, 2018, and contains forward-looking statements that involve risk and uncertainty. Actual results may differ materially from those set forth in such statements. For a discussion of these risks and uncertainties, you should review the Forward-Looking Statements Disclosure in the earnings press release we issued today as well as Adobe's SEC filings.During this call, we will discuss GAAP and non-GAAP financial measures. A reconciliation between the two is available in our earnings release and in our updated investor datasheet on Adobe's Investor Relations website.Call participants are advised that the audio of this conference call is being webcast live in Adobe Connect and is also being recorded for playback purposes. An archive of the webcast will be made available on Adobe's Investor Relations website for approximately 45 days and is the property of Adobe. The call audio and the webcast archive may not be re-recorded or otherwise reproduced or distributed without prior written permission from Adobe.I will now turn the call over to Shantanu.Shantanu NarayenThanks, Mike, and good afternoon. Q3 was a record quarter for Adobe. We delivered $2.29 billion in revenue, representing 24% year-over-year growth. GAAP earnings per share for the quarter was $1.34, and non-GAAP earnings per share was $1.73.Adobe is empowering people to create and transforming how businesses compete. Our execution against this strategy is driving strong financial results across our Digital Media and Digital Experience businesses. In every market around the world, students, creatives, enterprises and governments are choosing Adobe Creative Cloud, Document Cloud and Experience Cloud to deliver the transformative digital experiences required to compete and win today. In our Digital Media business, we achieved strong growth in both Creative Cloud and Document Cloud revenue in Q3. Net new Digital Media annualized recurring revenue or ARR was $339 million, and total Digital Media ARR exiting Q3 grew to $6.4 billion. Key Digital Media customer engagements in the quarter included the U.S. Department of Education, Facebook, Marks & Spencer and Walmart.Creative Cloud has become the creativity platform for all with millions of highly engaged subscribers and a strong base of trialists whom we actively convert each month into paying customers. Whether it's YouTubers looking for an intuitive video solution or mobile-first photography enthusiasts, we continue to see significant opportunities for growth in new customer segments as well as untapped potential in emerging markets. Video continues to be an explosive category. In June, we previewed Project Rush, a new video editing app that makes creating and sharing online video content easier than ever. Whether your passion is vlogging about food or posting a cool skateboarding clip, Project Rush gives users a way to create video projects across surfaces, providing them with maximum creative flexibility.This week at IBC, we shared a slate of new video creation capabilities that'll speed up video production and enable more seamless workflows for professional editors and animators. This includes Adobe Sensei-powered features for audio editing, color grading and animation in Premiere Pro, Audition, Character Animator and other video tools.Lightroom CC, our cloud-based photography service, continues to attract new customers. We announced a number of updates to Lightroom CC and Lightroom Classic for Mac, Windows, Android and iOS and shipped several improvements including new in-app learning capabilities, support for new cameras and more than 1,200 different lenses. We previewed a brand-new feature, Best Photos, which combines Adobe Sensei intelligence with user-made edits to quickly recommend the best photos within an album.Adobe XD, our all-in-one UX solution for designing and prototyping websites and apps, is quickly becoming the leader in the screen design category with strong monthly active usage among customers. This quarter, we unveiled new open platform capabilities, which allow users to customize their workflow with a broader ecosystem of community and partner plug-ins. As students around the world head back to school, Adobe is partnering with educators and institutions to ensure that creativity, a core 21st century skill, is a central part of curriculum and that students have access to the creative tools they need. Adobe Spark, our app for easily creating high-quality graphics, web pages and video stories, is a cornerstone of this effort. This quarter, we were proud to partner with the Ministry of Skill Development in India to enable more than 1 million students to access Spark.",
                 'AMD': "Laura Graves - Advanced Micro Devices, Inc.Thank you. And yes, welcome to AMD's third quarter 2018 conference call. By now, you should have had an opportunity to review a copy of our earnings release and slides. If you have not reviewed these documents, they can be found on the Investor Relations page of AMD's website, www.amd.com.Participants on today's conference call are Dr. Lisa Su, our President and Chief Executive Officer; and Devinder Kumar, our Senior Vice President, Chief Financial Officer, and Treasurer. This is a live call, and will be replayed via webcast on our website.I would like to highlight some important dates for you. AMD's next Horizon event is scheduled for Tuesday, November 6, 2018, where we will discuss innovation of AMD products and technologies, specifically designed for the datacenter on industry-leading 7-nanometer process technology. Dr. Lisa Su, President and Chief Executive Officer, will present at the Credit Suisse 22nd Annual Technology Media & Telecom Conference on Tuesday, November 27. And our 2018 fourth quarter quiet time will begin at the close of business on Friday, December 14.Today's discussion contains forward-looking statements based on the environment as we currently see it. Those statements are based on current beliefs, assumptions, and expectations; speak only as of the current date; and as such, involve risks and uncertainties that could cause actual results to differ materially from our expectations.We will refer primarily to non-GAAP financial measures during this call except for revenue, gross margin, and segment operational results, which are on a GAAP basis. The non-GAAP financial measures referenced today are reconciled to their most directly comparable GAAP financial measure in today's press release posted on our website. Please refer to the cautionary statements in our press release for more information. You will also find detailed discussions about our risk factors in our filings with the SEC and, in particular, AMD's Quarterly Report on Form 10-Q for the quarter ended June 30, 2018.Now, with that, I will hand the call over to Lisa. Lisa?Lisa T. Su - Advanced Micro Devices, Inc.Thank you, Laura, and good afternoon to all those listening in today. We executed well in the third quarter. We continued to build momentum for our new products as strong sales of our Ryzen and EPYC processors offset soft GPU channel sales, and drove our fifth consecutive quarter of year-on-year revenue growth, increased profitability and margin expansion.Third quarter revenue was $1.65 billion, an increase of 4% from a year ago. Looking at our Computing and Graphics segment, third quarter CG segment revenue increased 12% year-on-year, driven by significant growth in both client processor and OEM GPU sales that offset a larger-than-expected decline in channel GPU sales.Ryzen processor sales increased to more than 70% of our total client revenue in the quarter. We delivered our highest processor unit shipment in nearly four years, and believe we gained desktop and notebook client processor unit share in the quarter, driven by growth with both OEMs and in the channel. In desktop, we had strong demand for our higher end Ryzen 7, Ryzen 5 and Ryzen Threadripper processors, helping to drive a double-digit percentage year-over-year and sequential improvement in client processor ASP.We expanded our desktop offerings in the quarter, bringing our Zen processor core and Vega Graphics to the entry level part of the market with the Athlon APU, and launching our flagship 32-core Threadripper 2 processor. With these new introductions, we now have a top to bottom lineup of client processors, based on our high-performance Zen architecture.In notebooks, Ryzen mobile processor unit shipments doubled sequentially for the second straight quarter as OEMs ramped production of their latest AMD-based notebooks. 54 of the 60 Ryzen processor based notebooks planned for 2008 (sic) [2018] (04:45) have launched with the final notebooks expected to go on sale this quarter. Based on the success of first-generation Ryzen mobile notebooks, the expanded breadth of our customer engagements and our design win momentum, we are on track for an even larger assortment of AMD-powered notebooks in 2019.In graphics, the year-over-year revenue decrease was primarily driven by significantly lower channel GPU sales, partially offset by improved OEM and datacenter GPU sales. Channel GPU sales came in lower than expected, based on excess channel inventory levels, caused by the decline in blockchain-related demand that was so strong earlier in the year. OEM GPU sales in the third quarter increased by a strong double-digit percentage year-over-year as new design wins began to ramp, including first shipments of our mobile Vega GPUs to support new premium notebooks launching this quarter.",
                 'AAP': 'Elisabeth EislebenGood morning, and thank you for joining us to discuss our third quarter 2018 results. I’m joined by Tom Greco, our President and Chief Executive Officer; Jeff Shepherd, our Executive Vice President, Chief Financial Officer, Controller and Chief Accounting Officer; Bob Cushing, our Executive Vice President of Professional; and Mike Broderick, our Executive Vice President, Merchandising and Store Operations Support. Following their prepared remarks, we will turn our attention to answering your questions.Before we begin, please be advised that our comments today may include forward-looking statements as defined by the Private Securities Litigation Reform Act of 1995. While actual results may differ materially from those projected in such statements due to a number of risks and uncertainties, which are described in a Risk Factors section in the company’s filings with the Securities and Exchange Commission, we maintain no duty to update forward-looking statements made.Additionally, our comments today include certain non-GAAP financial measures. We believe providing these measures helps investors gain a more complete understanding of our results and is consistent with how management views our financial results. Please refer to our quarterly press release and accompanying financial statements issued today for additional detail regarding the forward-looking statements and reconciliations of these non-GAAP financial measures to the most comparable GAAP measures referenced in today’s call. The content of this call will be governed by the information contained in our earnings release and related financial statements.Now, let me turn the call over to Tom Greco.Tom GrecoThanks, Elizabeth, and good morning. First, I’d like to thank the entire Advance team and our network of Carquest Independent for a strong quarter, characterized by progress on virtually every financial and marketplace metric. It’s because of their dedication and unrelenting focus on the customer that we were able to deliver our strongest comparable sales growth in nearly eight years, which in turn led to improved market share performance. It’s clear that our transformation actions are beginning to take hold, driving meaningful improvements in execution, enabling us to earn more business with our valued customers.In the third quarter, net sales increased by 4.3% to $2.3 billion and comp sales were up 4.6%. Our adjusted operating income margin of 8.5% increased 62 basis points compared to the prior year quarter. And our adjusted earnings per share increased 32.2% to $1.89.Year-to-date, our free cash flow was $576 million, an increase of $336 million year-over-year, as we continue to drive disciplined cash management practices throughout the organization. Our sales performance in the quarter is a testament to the hard work of our entire team, truly living our cultural belief and executing on our mission, Passion for Customers, Passion for Yes!Not only did we deliver the strongest comparable sales growth since 2010, our absolute growth in the quarter was above the industry average, driven by increased units per transaction and average ticket value. Importantly, we strengthened our customer value proposition and delivered balanced, consistent growth throughout Q3 in both our DIY omni-channel and professional businesses.From a geographic perspective, we saw improvement across all 12 regions. Our Southeast, Midwest, Appalachia, Northeast, and Mid-Atlantic regions led the way. From a category perspective, we saw increased sales across nearly every category with the strongest growth in brakes, batteries, and optics, where we delivered high single-digit growth. Without a doubt, we’re seeing better coordination and planning across our merchandising, marketing and field operations teams, which is also contributing to our momentum.Finally, we’re leveraging supplier partnerships and improved analytical capabilities to ensure we have the right inventory, in the right locations, at the right time. This is a work-in-progress and we still have significant opportunity, but we believe that the improvements made so far are helping our improved execution.Going a little deeper in Professional, we saw improved performance across every AAP banner in the U.S., including Advance, Worldpac, Autopart International and importantly both Carquest Corporate and Independent stores. Our customers want choices, and thanks to cross-banner visibility, we’re doing a much better job of leveraging AAP’s industry leading assortment of brands to ensure our pro customers have the options they desire to meet the needs of their customers.Through continued focus on our customer value proposition and direct feedback from our professional customers, we recently launched myadvance.com. This is an interactive, easy to use, mobile friendly platform where we’ve combined multiple online tools and capabilities into one place, including our Advance Pro online ordering platform.On myadvance.com, professional customers can access industry-leading cross-banner assortment as well as best-in-class shop services and solutions such as an expert corner featuring content rich training and shop business solution advice from automotive aftermarket industry leaders. These initiatives are driving incremental growth for our Professional business.We’re equally excited about the growth within our DIY omni-channel business. In Q3, we saw increases in both DIY retail and DIY online, marking the second consecutive quarter with sales growth in both. We launched our new advertising campaign, Think Ahead. Think Advance, and it’s off to a great start. We continue to invest in e-commerce, where enhancements we made to our website improved the customer experience, including reduced page load times and additional personalization modules.In addition, well today, the majority of our online orders are, buy online pickup in store, we’ve made progress on our ship to home capabilities. Overall, our investments drove strong double-digit revenue growth in Q3. While we are pleased that our improving execution is driving topline strength, we remain focused on controlling expenses across the organization. I was delighted with our store team’s ability to once again, leverage additional customer service hours through increased sales in the third quarter.In addition, we’re seeing a noticeable shift in culture as safety has become a critical focus throughout Advance. Importantly, our progress on safety initiatives enabled a reduction in liability and vehicle claims in the quarter and we expect to see further benefits as a result of our safety initiatives. Our investment in this critical area has paid off quickly and I’m confident we’re developing a true safety culture with heightened attention on team member safety throughout Advance.',
                 'AES': "I would now like to turn the conference over to Ahmed Pasha, Vice President, Investor Relations. Please go ahead.Ahmed PashaThank you, Brendon. Good morning and welcome to our third quarter 2018 financial review call. Our press release, presentation and related financial information are available on our website at aes.com.Today, we will be making forward-looking statements during the call. There are many factors that may cause future results to differ materially from these statements. Please refer to our SEC filings for a discussion of these factors.Joining me this morning are Andrés Gluski, our President and Chief Executive Officer; Tom O'Flynn, our Chief Financial Officer; Gustavo Pimenta, Deputy Chief Financial Officer; and other senior members of our management team.With that, I will turn the call over to Andrés. Andrés.Andrés GluskiGood morning everyone and thank you for joining our third quarter 2018 financial review call. We had a strong quarter demonstrated by solid financial results and excellent progress towards achieving our strategic goals. We continue to improve the returns from our existing portfolio and position AES for long-term sustainable growth.Our third quarter adjusted EPS of $0.35 puts us at $0.88 for the first nine months of 2018 which is 35% higher than the $0.65 we earned in the same period last year. We remain on-track to achieve our 2018 guidance and longer term expectations. Tom will discuss our results in more detail after I provide a review of our strategic accomplishments. I will structure my remarks today around four overall themes; first, optimizing our returns; second, our growing backlog of renewable projects; third, advancing our LNG strategy; and finally, deploying new technologies. I have discussed these themes in the past, and on this call I will provide concrete examples of how we're delivering on each in support of our overall strategy.Beginning on Slide 4 with optimizing our returns. We have been reshaping our portfolio to deliver attractive returns to our shareholders while reducing our overall risk and carbon footprint. As can be seen on the slide, our renewable investments are projected to produce low-to-high teen IRRs across all markets assuming conservative terminal values. Specifically, as you may recall, we bought sPower in 2017 at a high single-digit return; since then we have taken steps to enhance that return including refinancings and operational improvements. This morning we announced that we have agreed to sell 24% of sPower's operating portfolio to Ullico. As a result of all of these actions we have improved our expected return on sPower's operating portfolio to around 13%, and we will use the proceeds to help fund new solar and wind projects in the U.S.Now turning to our backlog of renewable projects beginning on Slide 5. Our robust pipeline continues to increase driven by our focus on select markets where we can take advantage of our global scale and synergies with our existing businesses. So far this year we have signed 1.9 gigawatts of long-term PPAs for renewable projects or 93% of our internal projection of 2 gigawatts for full year 2018. We are on pace to sign 2 to 3 gigawatts of new PPAs annually for 2019 and 2020. We expect this capacity to be split 50-50 between the U.S. and internationally, and similarly, between solar and wind. By the end of 2020 we expect to have signed 7.5 gigawatts of new renewable PPAs all of which will be online by 2022. To complement our strategy to invest in attractive renewable projects and expand our environmental, social and governance related disclosures; next week we will be releasing a climate scenario report that complies with the guidelines of the task force on climate related financial disclosures and includes updated carbon intensity reduction targets that reflect our renewable growth.Now onto Slide 6 and how we are capitalizing on our existing footprint. As you may recall, on our last call we introduced our green blend and extend strategy. With this win-win strategy we leverage our existing platforms, contracts and relationships to negotiate new long-term PPAs with higher returns than we would otherwise get through a bidding process. We see potential opportunities to execute on this strategy across many of our markets including Chile, Mexico and United States. In the near-term we see an addressable universe of 7 gigawatts across our portfolio which could substantially increase as other markets capitalize on the economic benefits of renewables.Turning to Slide 7; we have signed two green blend and extend contracts for a total of 576 megawatts in Chile and Mexico. The contract in Chile was the first of it's kind; we signed an 18-year contract with an existing customer for 1,100 gigawatt hours of annual delivery [ph] which is equivalent to 270 megawatts of renewable capacity. This will lengthen AES Gener's average contract life to 11 years, replace 5% of AES Gener's total load, and 40% of the thermal PPAs expiring in 2022. We are also implementing a similar contract in Mexico with our off-taker Penoles. To help Penoles gradually replace pet-coke with greener, efficient renewable energy, we negotiated a 25-year PPA to build the 306 megawatt Mesa La Paz wind project leveraging our strong existing customer relationship and our global renewables capabilities. This will increase our average contract life with Penoles from 8 years to 17 years.",
                 'AMG': "I would now like to turn the conference over to your host, Mr. Jeff Parker, Vice President, Investor Relations for AMG. Thank you. You may begin.Jeffrey ParkerThank you for joining AMG to discuss our results for the third quarter of 2018.In this conference call, certain matters discussed will constitute forward-looking statements. Actual results could differ materially from those projected due to a number of factors, including, but not limited to, those referenced in the Company's Form 10-K and other filings we make with the SEC from time to time. We assume no obligation to update any forward-looking statements made during the call.AMG will provide on the Investor Relations section of its website, at www.amg.com, a replay of the call, a copy of our announcement of our results for the quarter, a reconciliation of any non-GAAP financial measures to the most directly comparable GAAP financial measures, including a reconciliation of any estimates of the Company's economic earnings per share for future periods that are announced on this call and an investor presentation. AMG encourages investors to consult the Investor Relations section of its website regularly for updated information.With us on the line to discuss the Company's results for the quarter are Nate Dalton, President and Chief Executive Officer; and Jay Horgen, Chief Financial Officer.With that, I'll turn the call over to Nate.Nate DaltonThanks, Jeff, and good morning everyone. Against the challenging industry backdrop, AMG generated solid results in the third quarter of 2018, including positive net client cash flows of approximately $1 billion and economic earnings per share of $3.45. Our results reflect the diversity of our global business and our strategic position in attractive, return-oriented products, where we continue to see significant client demand for our affiliate strategies and the distinctive investment return streams they create.In terms of third quarter flow, we saw continued strong demand for alternatives, which included another quarter of very significant illiquid product fundraising from institutional clients, partially offset by softness in liquid alternatives in the retail channel, where investors tend to be more sensitive to short-term performance. The ongoing strength of our alternative products was partially offset by net outflows from equity strategies, driven primarily by emerging markets products and U.S. equity retail outflows.At the highest level, we were pleased with our positive organic growth in the third quarter, despite industry headwinds. We benefited from having built a diverse global business and we're pleased to see a very broad array of product types. Affiliates and geographies contributed significantly to our flow composition, as we saw strong sales across Baring Asia, BlueMountain, Capula, EIG, Pantheon, and PFM within our alternative category. [Indiscernible] Frontier and Harding Loevner were notable contributors within our equities category.Now, looking ahead, I want to address the current market environment and how we have built and continue to position our business. Obviously, it's been a volatile period. Global and U.S. equity markets have declined roughly 10% month to date as of Friday. And frankly that understates some of the underlying volatility we've been seeing. Anecdotal evidence points to some de-risking in this environment and you can see that reflected in this month's retail flow data.In the short run, this obviously creates challenges in various segments of the asset management industry. But in terms of AMG, we've been evolving the business in anticipation of more volatile and fundamentals-driven market, with the highest quality active managers proving their worth. Volatile markets underscore the need for investors to diversify their equity and fixed income exposure, especially when markets are at elevated levels. Finally, within both traditional and alternative categories, volatile markets create environment with the best active managers, like our affiliates, who can significantly outperform their benchmarks in simple passive products.Looking ahead, I would focus on three aspects of our business and strategic position. One, the diversity of the business we've built over the years. Two, the opportunities for the best active managers to outperform in this environment. And three, the quality of our affiliates and the distinctive return streams they produce.First, on the diversity of our business. Over the last decade, we have deliberately doubled the proportion of our business in alternatives from approximately 20% in the beginning of 2008 to approximately 40% of assets under management today, spread across a very diverse set of high quality alternative products, while increasing our exposure to growing parts of client portfolios at the highly active alpha end of the barbell.This evolution has come from both investments in new affiliates, as well as product innovation and development by us and our affiliates. Our alternatives business now has approximately $320 billion in assets under management, making AMG one of the largest alternative managers with one of the broadest and most diverse ranges of liquid and illiquid alternative strategies, managed by leading investors, including AQR, Baring Asia, BlueMountain, Capula, EIG, First Quadrant, Pantheon, PFM, Systematica, ValueAct and Winton.In addition, our substantial exposure to uncorrelated alternative strategies should increase the stability and resilience of our business across market cycles, while most importantly, proving attractive to clients, so increasing the long-term organic growth potential of our business.",
                 'AFL': "I would now like to turn the call over to Mr. David Young, Vice President of Aflac Investor & Rating Agency Relations.David Young - Aflac, Inc.Thank you and good morning. Welcome to our third quarter call. This morning, we will be hearing remarks from Dan Amos, Chairman and CEO of Aflac Incorporated, about the quarter as well as our operations in Japan and the United States. Then, Fred Crawford, Executive Vice President and CFO of Aflac Incorporated, will follow with more details about our financial results, outlook and capital management. We will then open our call to questions.Joining us this morning during the Q&A portion are members of our executive management team in the U.S., Teresa White, President of Aflac U.S.; Eric Kirsch, Global Chief Investment Officer; Rich Williams, Chief Distribution Officer; Al Riggieri, Global Chief Risk Officer and Chief Actuary; and Max Brodén, Treasurer.We are also joined by members of our executive management team in Tokyo at Aflac Life Insurance Japan, Charles Lake, President of Aflac International and Chairman, Representative Director; Masatoshi Koide, President and Representative Director; Todd Daniels, Director and Principal Financial Officer; and Koji Ariyoshi, Director and Head of Sales and Marketing.Before we start, let me remind you that some statements in this teleconference are forward-looking within the meaning of federal securities laws. Although we believe these statements are reasonable, we can give no assurance that they will prove to be accurate because they are prospective in nature.Actual results could differ materially from those we discuss today. We encourage you to look at our Annual Report on Form 10-K for some of the various risk factors that could materially impact our results. The earnings release is available on the Investors page of Aflac's website at investors.aflac.com and includes reconciliations of certain non-GAAP measures.I'll now hand the call over to Dan.Daniel P. Amos - Aflac, Inc.Thank you, Dave, and good morning, and thank you for joining us. Let me begin by saying that the third quarter 2018 concluded a great nine months for Aflac and well positioned us to achieve the goals we set for the year. As you saw from the earnings release yesterday, I am pleased that we expect to come in at the high end of the upwardly revised 2018 adjusted EPS outlook. Fred will provide more details in his comments shortly.Aflac Japan, our largest earnings contributor, generated strong financial results. In yen terms, Aflac Japan's pre-tax profit margin was ahead of expectations both for the quarter and for the first nine months.Aflac Japan's third sector sales results of a 2.6% decrease was consistent with our expectations. This reflects sales growth in our new cancer insurance and a decline in our medical insurance sales. As medical sales came-off a strong year bolstered by refresher core products, our distribution turned its focus this year on our new cancer product as they tend to do when a new core product is introduced. We expect similar results in the fourth quarter and continue to anticipate third sector new sales growth for the year to be in the low-single-digit.As I've said many times, our focus is on defending and growing our leading third sector franchise. We are indifferent as to the mix of medical and cancer sales as long as we are satisfying the needs of the consumer and our distribution partners. Regarding distribution, we had meaningful production across all channels. Traditional agencies have been and remain vital to our success.Our Alliance partners has also had significant contribution to our sales results, with such an extensive distribution network, including Japan Post 20,000-plus postal outlets selling our cancer insurance, we are solidifying our goal to be where people want to buy insurance.Our focus remains on remaining on maintaining our leadership position in the sale of third sector products that are less interest rate-sensitive and have strong and stable margins. We will continue to refine our existing product portfolio and introduce innovative new third sector products to maintain our market leadership.Turning to Aflac U.S., we are pleased with our financial performance. The pre-tax profit margins exceeded our expectations both for the quarter and for the first nine months. Our third quarter new annualized premium sales, together with our sales outlook, keep us on track to achieve the lower end of our anticipated 2018 new annualized premium sales growth of the 3% to 5% increase.As you think about U.S. sales, keep in mind that Aflac is different from our peers and that the majority of our sales come from independent sales agents. We are fortunate to have such a strong independent field force which is truly unique within our industry. These career sales agents are best positioned within the industry to reach and therefore succeed with smaller employers and groups with fewer than 100 employees.Aflac's independent career agents have been the driving force behind Aflac's ability to dominate the smaller case market. And I continue to believe this market is ours to grow. We continue to expect higher growth in broker sales. Our team of broker sales professionals has made great strides in successfully strengthening Aflac's relationship within the large broker community.",
                 'A': "It is now my pleasure to hand the conference over Ms. Alicia Rodriguez, Vice President Investor Relations. Ma’am, you may begin.Alicia RodriguezThank you, Brian, and welcome, everyone, to Agilent’s third quarter conference call for fiscal year 2018. With me are Mike McMullen, Agilent’s President and CEO; and Didier Hirsch, Agilent’s Senior Vice President and CFO.Joining in the Q&A after Didier’s comments will be Jacob Thaysen, President of Agilent’s Life Sciences and Applied Markets Group; Sam Raha, President of Agilent’s Diagnostics and Genomics Group; and Mark Doak, President of the Agilent CrossLab Group.I’m also please to announce that Bob MacMahon is joining us on the call today as well. As you know, he will be taking on the role as Agilent’s CFO in September due to Dider’s retirement at the end of October. You can find the press release and information to supplement today’s discussion on our website at www.investor.agilent.com. While there, please click on the link for financial results under the Financial Information tab. You will find an investor presentation along with revenue breakouts and currency impacts, business segment results and historical financials for Agilent’s operations. We will also post a copy of the prepared remarks following this call.Today’s comments by Mike and Didier will refer to non-GAAP financial measures. You will find the most directly comparable GAAP financial metrics and reconciliations on our website. Unless otherwise noted, all references to increases or decreases in financial metrics are year-over-year. References to revenue growth are on a core basis. Core revenue growth excludes the impact of currency and acquisitions and divestitures within the past 12 months. Guidance is based on exchange rates as of July 31.–––We will also make forward-looking statements about the financial performance of the company. These statements are subject to risks and uncertainties and are only valid as of today. The company assumes no obligation to update them. Please look at the company’s recent SEC filings for a more complete picture of our risks and other factors.And now, I’d like to turn the call over to Mike.Michael McMullenThanks, Alicia. Hello, everyone. Thanks for joining us on today's call. Before I discuss the Q3 financial highlights and our updated outlook I'm pleased to have Bob McMahon join the call. Bob is an excellent choice for Agilent’s next CFO and a very capable successor to Didier. Bob brings a strong track record of leadership to our team. Many of you already know Bob from his previous role of CFO of Hologics. He officially assumes the CFO role beginning September 1.As Didier hands off the baton, he will serve as adviser capacity until his retirement at the end of October. Bob and Didier working together to ensure a smooth transition. I first met Bob when in Palo Alto where we shared our perspective on business and company culture. We had an important conversation about values and the importance of business. I knew immediately that Bob would be a great fit for the Agilent culture and of course his management style and business acumen are a perfect match for our approach to trading shareholder value. Bob has joined Agilent in an exciting time. I'm confident he'll help us lead the next phase of Agilent growth.While I'm very excited to have Bob join the Agilent team I will greatly miss Didier's partnership and counsel. He has played a key role in the transformation of the company and our excellent business results. It's important for the CEO to have a very capable CFO. I couldn't have asked for a better partner. So thank you Didier. You will be missed by me and our Agilent team.Now, let me turn to our Q3 financial performance. The Agilent team delivered another strong quarter with both growth and earnings exceeding our expectations. Our core revenue grew 6% and is above the high-end of our guidance. Our adjusted EPS of $0.67 is $0.04 above the high-end of our guidance despite currency headwinds since our last guide. This is a 14% increase from a year-ago. We delivered an adjusted operating margin of 22.6%, which is an increase of 110 basis points from a year-ago. This marks our 14th consecutive quarter of improving our core operating margins.Let's take a closer look at our results by our end-markets. We continue our strong Pharma performance with 8% core growth. This is against a tough compare as we grew 10% in Q3 '17. We see strength across all our business groups with particularly strong performance in mass spectrometry, Cell Analysis, CrossLabs consumables and services and genomics. Growth remains robust in both the biopharma and small molecule market segments.Our Chemical and Energy market revenue grew 12%. We are quite pleased with this strong growth. Again, against a difficult prior-year compare of 10%. Ongoing market investment remains positive. This is in spite of tariff rhetoric and retaliatory policies you've been hearing in the news. From a product perspective strength in spectroscopy, GC, CrossLabs consumables and services is driving this result. Geographically strong gains in China and Europe are leading the overall global growth.Revenue grew 3% in academia and government inline with expectations, strong performance from Cell Analysis, molecular spectroscopy, ICP/MS and CrossLabs consumers and services are driving the results. China and the rest of Asia are delivering double-digit growth in this end-market.",
                 'APD': "Beginning today's call is Mr. Simon Moore, Vice President of Investor Relations. Please go ahead, sir.Simon R. Moore - Air Products & Chemicals, Inc.Thank you, Vicky. Good morning, everyone. Welcome to Air Products third quarter 2018 earnings results teleconference. This is Simon Moore, Vice President of Investor Relations. I'm pleased to be joined today by Seifi Ghasemi, our Chairman, President and CEO; Scott Crocco, our Executive Vice President and Chief Financial Officer; and Sean Major, our Executive Vice President, General Counsel and Secretary. After our comments, we'll be pleased to take your questions.Our earnings release and the slides for this call are available on our website at airproducts.com. Please refer to the forward-looking statement disclosure that can be found in our earnings release and on slide number 2. Now, I'm pleased to turn the call over to Seifi.Seifollah Ghasemi - Air Products & Chemicals, Inc.Thank you, Simon, and good morning to everyone. Thank you for joining us on our call today. We certainly do appreciate your interest in Air Products. The talented, committed, and motivated team at Air Products delivered another excellent set of safety and financial results. Our record adjusted earnings per share of $1.95 is up 18% versus last year. This is the 17th consecutive quarter that we have reported year-on-year EPS growth and the fifth consecutive quarter we have delivered year-on-year EPS growth of more than 15%. We continue to be the safest and most profitable industrial gas company in the world with a record quarterly dividend margin of over 36%.We continue to generate significant cash, which supports our robust dividend policy. And we do have the strongest balance sheet in the industry, which gives us the ability to commit a significant amount of capital to grow Air Products in the coming years. And most important, we have a great team of hardworking, dedicated, talented, and committed people at Air Products, who have stayed focused on working hard every day to serve our customers and create value for our shareholders.Now, please turn to slide number 3. We continue to improve our safety results with a reduction of 67% in our lost time injury rate and a reduction of 52% in our recordable injury rate. These results can only happen when all of our 15,000 employees around the world are committed to safety and continuous improvement.On slide number 4, you can see our goal for the company, to be the safest, most diverse and most profitable industrial gas company in the world, providing excellent service to our customers. And I want to emphasize that the most diverse in our goal refers to our people. We value a diverse workforce.Now, please turn to slide number 5. You can see our overall management philosophy that we have talked to you about many times over the last four years. But it is worth repeating because we continue to be focused on shareholder value, capital allocation, and an empowered and decentralized organization.Now, please turn to slide number 6. It was almost four years ago that I shared our original Five-Point Plan: we have successfully focused on Air Products' core industrial gases business; we have restructured the organization; changed the culture; controlled capital and costs; and aligned our rewards. We have done what we promised to do.Now, please turn to the slide number 7. Our journey is never complete, and it's time to evolve our Five-Point Plan to guide us over the coming years. Let me explain each of the points on this slide one at a time. First, from the left side, in terms of sustaining our lead, we will keep our focus on safety. We have done well, but even one injury is too many. Accidents don't happen by themselves. Every incident or accident is preventable. We want to be the best-in-class in everything we do. We need to be the best-in-class operationally, for example, to make sure our plants are running all the time. The same thing with our human resource processes, financial processes, safety processes. Everything that we do, we should aim to, and we will be, the very best in the industry. And productivity, we obviously need to continue to focus on productivity to maintain our margins.Second, in terms of deployment of capital, as you have heard me say before, we believe that we have at least $15 billion of capital to commit over the next five years. This includes cash and debt capacity available today and the investable cash flow we expect to generate in the next five years. I remain very confident – and I like to repeat this – I remain very confident that we will be able to commit the full $15 billion to very high-quality industrial gas projects over the next five years.",
                 'AKAM': "I would now like to introduce your host for today’s conference Tom Barth, Head of Investor Relations. You may begin.Tom BarthThank you, Gigi. Good afternoon, everyone, and thank you for joining Akamai's third quarter 2018 earnings conference call. Speaking today will be Tom Leighton, Akamai's Chief Executive Officer; and Jim Benson, Akamai's Chief Financial Officer.Before we get started, please note that today's comments include forward-looking statements, including statements regarding revenue and earnings guidance. These forward-looking statements are subject to risks and uncertainties and involve a number of factors that could cause actual results to differ materially from those expressed or implied by such statements.Additional information concerning these factors is contained in Akamai's filings with the SEC, including our Annual Report on Form 10-K and Quarterly Reports on Form 10-Q. The forward-looking statements included in this call represent the Company's view on October 29, 2018. Akamai disclaims any obligation to update these statements to reflect future events or circumstances.As a reminder, we will be referring to some non-GAAP financial metrics during today's call. A detailed reconciliation of GAAP and non-GAAP metrics can be found under the financial portion of the Investor Relations section of akamai.com.And with that, let me turn the call over to Tom.Thomson LeightonThanks, Tom, and thank you all for joining us today. Akamai delivered excellent results in the third quarter. Revenue was $670 million, up 7% over Q3 of last year and up 8% in constant currency, with continued very strong performance for our security products and continued improvement in our Media and Carrier Division. Q3 non-GAAP EPS was $0.94 per diluted share, up 47% year-over-year. This very strong result was driven by our solid revenue growth, the impact of cost reductions made over the past year and a lower tax rate.EBITDA margins in Q3 improved to 41%, and non-GAAP operating margins improved to 27%. This mark the fourth consecutive quarter of improving margins and we expect further improvements by the end of the year. Looking further ahead, we are now confident that we can achieve non-GAAP operating margins of 30% in 2020, while continuing to invest in innovation and new products to drive our future growth.In Q3, our security portfolio was again the fastest growing part of our business, with revenue of $169 million, up 39% over Q3 of last year in constant currency. Our security business accounted for 25% of our total Q3 revenue and exited the quarter at a run rate of nearly $700 million per year, making Akamai one of the world's largest cloud security providers.Our security products received accolades from two leading analyst firms last month. Gartner named Akamai as a leader in its Magic Quadrant for Web application firewalls for the second year in a row. And Forrester named Akamai as a leader in its new wave for Bot Management, giving Akamai’s our highest rating possible for attack response, threat research, reporting and analysis, roadmap and market approach.In a world where lots of claims are made by competing companies, it's gratifying to receive this level of recognition from the world's leading industry analyst firms. There are several reasons why Akamai is a leader in Cybersecurity. Our track record of innovation and emphasis on R&D combined with smart and accretive acquisitions. Our edge platforms enormous capacity and close proximity to users and devices, the insights in real-time data we gained from protecting so many of the world's leading enterprises, and our team of experts who provide exceptional customer service and support.Many of our customers have told us that they simply have too much at stake to risk their business on anything less. Most of our security revenue today is based on protecting public-facing websites and applications from denial of service and application-layer attacks. We've also been developing novel products to protect internal enterprise applications. And I am very pleased to report that these new products have been gaining traction.Bookings in Q3 for our Enterprise Application Access and Enterprise Threat Protector services were up 30% over Q2 and up more than fourfold over Q3 of last year. Our new enterprise security customers include leading companies such as one of the largest banks in the UK, a $10 billion travel company, a $4 billion luxury fashion brand, two of the world's leading consulting firms and a $30 billion media company.We believe that our new enterprise security offerings are gaining traction because they provide the core capabilities needed for a zero-trust security architecture. It's called zero-trust because enterprises can no longer assume that any device or employee on their internal network is trustworthy, as because employees sometimes click on the wrong link and accidentally import malware, which can then lead to a very costly data breach.As a result, we've seen growing interest in new architectures that protect internal applications by granting access at the application layer instead of the network layer. This approach avoids exposing corporate networks through a VPN and eliminates the need to manage complex network firewall rules.",
                 'ALK': "I would now like to turn the call over to Alaska Air Group's Director of Investor Relations, Matt Grady.Matt Grady - Alaska Air Group, Inc.Thanks, Amber. Good morning, everyone, and thank you for joining us for our third quarter 2018 earnings call. On the call today, our CEO, Brad Tilden, will provide an overview of the business; Andrew Harrison, our Chief Commercial Officer, will share an update on our revenue results and outlook; and our CFO, Brandon Pedersen, will discuss our cost performance and cash flows. Several other members of our management team are also on hand to help answer your questions.Earlier this morning, Alaska Air Group reported third quarter GAAP net income of $217 million. Excluding merger-related costs and mark-to-market fuel hedging adjustments, Air Group reported adjusted net income of $237 million and adjusted earnings per share of $1.91, ahead of the First Call consensus.As a reminder, our comments today will include forward-looking statements regarding our future expectations which may differ significantly from actual results. Information on risk factors that could affect our business can be found in our SEC filings. On today's call, we will refer to certain non-GAAP financial measures such as adjusted earnings and unit costs excluding fuel. And as usual, we have provided a reconciliation between the most directly comparable GAAP and non-GAAP measures in our earnings release.And with that, I will turn the call over to Brad.Bradley D. Tilden - Alaska Air Group, Inc.Thanks, Matt, and good morning, everyone. We are rapidly approaching the two-year mark since our merger closed and while there's still a lot of work ahead of us, we are extremely happy about the pace of progress and we remain confident in our future. We've now completed roughly 90% of the integration milestones and have done so at a pace that is equal to or more rapid than virtually any other merger in the industry.The performance of our core business remains strong and our brand and product are gaining increasing traction in California. As our network investments mature and the merger synergies accelerate, we're doubling down on the disciplines of productivity, teamwork and cost control, disciplines that have been the source of our competitive advantage for many years now.In the third quarter, our financial performance started to turn the corner with our employees delivering unit revenues and unit costs that were both better than plan. We were especially encouraged to see RASM stabilize in the face of multiple headwinds. We expect RASM to inflect positive in the fourth quarter as we begin the steady climb toward the higher margins and higher returns that we believe are achievable with our combined network.As we've discussed, we're not satisfied with our current financial returns. Fuel prices continue to rise and we need to do more to recover these higher costs. One way we've responded is by raising bag fees for the first time in five years, bringing our fees into line with those that are now prevalent in the marketplace. We announced these changes last week and Andrew will discuss them further in just a moment.Stepping back a bit, we see an industry environment that has not yet adequately adjusted to higher fuel prices. We've slowed our growth in 2019, as market dynamics simply don't argue for adding much new capacity. That said, we are comfortable with our current growth and we're very comfortable with our new network footprint.We have a solid plan in place to leverage the many factors within our control and to drive unit revenues and margins higher going forward. We're making it even more granular as we go through the budget process. We're confident that we're on a path to significantly improve our profitability and our free cash flow in both 2019 and 2020.Before I discuss future plans, I would like to highlight a few of our team's accomplishments in the third quarter. First, we were proud to win Best U.S. Airline from both Condé Nast Traveler and KAYAK. The Condé Nast recognition is especially meaningful, as it continued a tradition. Virgin America had won this award for the past 10 years and now our folks have extended the streak to 11.",
                 'ALB': "David Ryan - Albemarle Corp.Thank you, and welcome to Albemarle's third quarter 2018 earnings conference call. Our earnings were released after the close of the market yesterday and you'll find our press release, earnings presentation and non-GAAP reconciliations posted to our website under the Investors Section at www.albemarle.com. Joining me on the call today are Luke Kissam, Chairman and Chief Executive Officer; Scott Tozier, Chief Financial Officer; Raphael Crawford, President, Catalysts; Netha Johnson, President, Bromine Specialties; and Eric Norris, President, Lithium.As a reminder, some of the statements made during the conference call about our outlook, expected company performance, production volumes and commitments, as well as lithium demand may constitute forward looking statements within the meaning of federal securities laws. Please note the cautionary language about forward-looking statements contained in our press release. That same language applies to this call.Please also note that some of our comments today refer to financial measures that are not prepared in accordance with GAAP. A reconciliation of these measures to GAAP financial measures can be found in our earnings release and the appendix of our earnings presentation, both of which are posted on our website.Now, I will turn the call over to Luke.Luke C. Kissam - Albemarle Corp.Thanks, Dave, and good morning, everyone. We apologize for the little snafu at the very beginning. Apparently, you could hear us, but we couldn't hear the operator; so, we apologize and hope we got that fixed.Turning to the third quarter, in the third quarter, our revenue grew $50 million, or 7%, and adjusted EBITDA grew by $36 million, or 18%, compared to the third quarter of 2017 excluding divested businesses. This marks our eighth consecutive quarter of double-digit adjusted EBITDA growth.During the third quarter, Bromine Specialties and Catalysts both reported pro forma adjusted EBITDA growth over last year. In Lithium, third quarter pricing increased year-on-year as expected; however, unexpected outages at three of our manufacturing sites during the quarter caused volume shortfalls, which resulted in our not being able to meet the sales commitments in the quarter. These issues were one-time in nature and have been addressed. All of our Lithium facilities are now running at forecasted production rates.In addition, our Lithium capital projects remain on track. We completed the tie-ins at La Negra II and expect to operate that unit at full rates in 2019. La Negra III and IV, which is a 40,000 met ton carbonate expansion, is progressing as planned towards commissioning during 2020. Earlier this year, we commissioned an expansion of our evaporation system in the Salar de Atacama, enlarging our evaporation pond capacity by 60%. Additional ponds of more than 450 acres are on schedule for completion in early 2019. This expanded pond system will provide sufficient feedstock for all of our carbonate production facilities in Chile.In China, we have completed all pre-commissioning activities at Xinyu II and are now transitioning that unit over to operations. We have begun startup activities and we'll be in this phase over the next few months. We expect significant hydroxide volumes from this unit in 2019. With respect to the Kemerton Lithium Hydroxide facility, we are on track to obtain all necessary approvals to begin earthwork at the site in December.Now, I'll turn the call over to Scott.Scott A. Tozier - Albemarle Corp.Thanks, Luke, and good morning, everyone. For the third quarter, we reported net income of $130 million or $1.20 per diluted share. Adjusted earnings per share were $1.31, an increase of about $0.31 per share compared to third quarter 2017, or 31% growth excluding divested businesses and other one-time items. Our businesses and lower corporate expenses delivered about $0.31 per share of growth with strong performance in Bromine Specialties and Catalysts. Our share repurchase program contributed about $0.04 during the quarter. Those gains were partially offset by a net cost increase in other areas, primarily due to a higher effective tax rate and unfavorable foreign exchange compared to the third quarter 2017.Let me talk about each of the businesses. Lithium reported third quarter net sales of $271 million and adjusted EBITDA of $114 million, each up about 1% year-over-year. Adjusted EBITDA margin was 42%. Average prices were up 6% versus the third quarter 2017 with battery grade salts providing most of that improvement. The gains were mostly offset by the volume impact from the unexpected plant outages. Hydroxide volumes and cost were unfavorable due to a one-week shutdown in Kings Mountain as a result of Hurricane Florence in North Carolina.",
                 'ARE': "I would now like to turn the conference over to Ms. Paula Schwartz with Investor Relations. Ms. Schwartz, please go ahead.Paula SchwartzThank you, and good afternoon, everyone. This conference call contains forward-looking statements within the meaning of the federal securities laws. The Company's actual results might differ materially from those projected in the forward-looking statements. Additional information concerning factors that could cause actual results to differ materially from those in the forward-looking statements is contained in the Company's periodic reports filed with the Securities and Exchange Commission.And now I'd like to turn the call over to Joel Marcus, Executive Chairman and Founder. Please go ahead, Joel.Joel MarcusThank you, Paula, and welcome everybody to the third quarter call for Alexandria. And with me today are Steve Richardson, Peter Moglia, Dean Shigenaga, Dan Ryan. The third quarter was an outstanding quarter by almost every financial and operating metric and particularly core operating metrics that were really stellar and Dean will talk more about that.My congratulations to our entire Alexandria family and for each person’s day-to-day, day in and day out operational excellence, which is what really makes it all happen. And also congratulations to the world-class accounting and finance team on their many years of hard work, resulting in our recent credit upgrade from Moody's, something very important.And Moody’s does focus on tenant quality and at Alexandria, it's one of our strongest characteristics of the Company. As you know from the earnings release, 52% of our annual rental revenue is from investment grade or large cap public companies. We are very proud that 79%, almost 80% of that revenue is from Class A assets in AAA locations, and about 60% of that is focused in Cambridge in San Francisco. Average lease term is about 8.6 years and 12.3 for top 20 tenants. So really a very strong and stellar core to the tenant base.I want to just make a couple of comments on the industry for the quarter. Venture funding in life science continued at very strong pace over $7 billion in the third quarter, marking the fourth consecutive quarter of over $5 billion invested. And this is really a record breaking trend driven by increase in deal size and well established venture firms raising larger funds and deploying capital at a faster pace.Something else that I think we are very fortunate, we had our 47th new drug approved on October 24, this year and the FDA has surpassed last year's 46th drug approval count and could be on pace to beat the all time record of 53 set in 1996. Strong bipartisan support resulted in legislation enacted to increase the National Institutes of Health. Overall funding, $2 billion to approximately $39.1 billion and we are very fortunate about that. I think that is one of the critical competitive advantages of the United States in the world of biomedical research.We did have very strong legislation, bipartisan legislation passed to address the opioid crisis signed in a law by the President, in the recent past. And certainly the opioid crisis has been a scourge resulting in the death of over 64,000 people last year greater than those died in the entire Vietnam War, which is actually hard to fathom. And so all of us in the industry and particularly ourselves are focused on specific things we can do to advance that project forward and we'll give you more details on that in coming quarters.Biotech IPO activity has been the strongest since 2014, approaching 50 certainly over the next couple of weeks and raising almost $5 billion during the first nine months. The NASDAQ biotech index has been down a bit recently due to the volatility this month with the markets as all of you know and we've seen certainly a flight of value and big cap safety.And I think with that, I'm going to turn it over to Steve to comment on a number of operational aspects, then Peter and then back.Stephen RichardsonGreat. Thank you, Joel. Steve Richardson here. This afternoon I'll highlight the continued strong demand for Alexandria’s highly differentiated Class A science and technology campuses and the country's leading innovation clusters.Building on what Joel had mentioned about the life science capital markets, we did have 17 IPOs this quarter and this is just part of an overall strong demand context, really driving stellar leasing and financial results with increases this quarter of 16.9% in cash and 35.4% in GAAP, the highest in 10 years.Drilling down to Alexandria’s cluster markets, it's important to know that Alexandria has been a first mover in each of these markets, and as such as a dominant position with the highest quality campuses immediately proximate to the country's leading life science research institutions.",
                 'ALXN': "Good morning, and welcome to the Alexion Pharmaceuticals, Incorporated Third Quarter 2018 Results Conference Call. Today's call is being recorded.For opening remarks and introductions, I would now like to turn the call over to Susan Altschuller. [Technical Difficulty] (00:17-00:28)Susan Altschuller - Alexion Pharmaceuticals, Inc.You can access the webcast slides that will be presented on this call by going to the Events section of our Investor Relations page on our website.Before we begin, I would like to point out that we will be making forward-looking statements, and these statements involve certain risks and uncertainties that could cause our actual results to differ materially. Please take a look at the risk factors discussed in our SEC filings for additional details. These forward-looking statements apply only as of today and we undertake no duty to update any of the statements after the call, except as required by law.I'd also like to remind you that we will be using non-GAAP financial measures, which we believe provide useful information for the understanding of our ongoing business performance. Reconciliations of our financial results and financial guidance are included in our press release. These non-GAAP financial measures should be considered in addition to, but not a substitute for, our GAAP results. Thank you.Ludwig?Ludwig N. Hantson - Alexion Pharmaceuticals, Inc.Thank you, Susan, and thank you all for joining us this morning. We continue to build on our momentum, advancing our mission of bringing hope to patients and families affected by rare diseases. We had an excellent third quarter with a number of significant achievements to highlight.Yesterday marked the anniversary of the FDA approval of Soliris for patients with MG. One year later, I'm very proud to say that we have delivered on our ambition of making MG the best Soliris launch yet.Last month, we reported remarkable topline results from our Phase 3 PREVENT study of Soliris in NMOSD. We're moving quickly to prepare global regulatory submissions, which can make Soliris the first approved therapy for patients with this devastating disease.Also last month, we announced an agreement to acquire Syntimmune and with its SYNT001, an innovative FcRn-targeted asset that holds great promise for treating IgG-mediated diseases. And once again, we delivered strong top and bottom line growth.Moving to slide six, we've continued to execute on our five key objectives throughout the year. We have made notable progress in addition to the achievements I've just mentioned. We continue to grow the in-line business with our Complement and Metabolic portfolios, delivering year-over-year revenue growth of 20% and volume growth of 26% in the quarter.MG remains a significant growth driver. And at the end of September, 560 MG patients were on therapy in the U.S.The PNH regulatory submissions for Ultomiris, formerly referred to as ALXN1210, have been accepted in the U.S., EU and Japan, and we're actively preparing for anticipated launches.I'm very pleased with the significant progress we've made in building our pipeline this year with Syntimmune, Wilson Therapeutics, Complement Pharma, and most recently, Dicerna. Our pipeline today is far more robust and diverse.Again, I will highlight the groundbreaking results from our Phase 3 study of Soliris in NMOSD, which showed a 94.2% reduction in the risk of adjudicated relapse and an adjudicated annualized relapse rate of 0.02.Finally, given our strong financial performance, we have increased our guidance. We have accomplished a lot in the first three quarters of this year: executing on our base business, successfully launching in MG, preparing for Ultomiris launches, and rebuilding our pipeline for long-term value creation, all while maintaining a focus on transforming the lives of patients with rare diseases.With that, I will now turn the call over to Paul to discuss the third quarter financial results and to provide details on our guidance. Paul?Paul J. Clancy - Alexion Pharmaceuticals, Inc.Thanks, Ludwig. I'm pleased to be reporting on another strong quarter. Starting with slide 8, we reported total revenues in the quarter of $1.27 billion, an increase of 20% year-over-year, driven by continued performance of Soliris and Metabolic portfolios. Our non-GAAP operating margin was 54% in the third quarter, an expansion of 916 basis points. This was well above our prior forecast, driven by topline leverage, timing of program spend and lower workforce costs.We're anticipating a lower operating margin in the fourth quarter of the year due to an expanding pipeline and sales and marketing investment to support our growth opportunities. I'll provide further details. Non-GAAP earnings per share growth was 40%, driven by topline performance and strong operating expense control.",
                 'ALGN': "I would now like to turn the conference over to your host, Shirley Stacy, VP of Corporate and Investor Communications.Shirley Stacy - Align Technology, Inc.Good afternoon, and thank you for joining us. I'm Shirley Stacy, Vice President of Corporate Communications and Investor Relations. Joining me for today's call is Joe Hogan, President and CEO; and John Morici, CFO. We issued third quarter 2018 financial results today via GlobeNewswire, which is available on our website at investor.aligntech.com.Today's conference call is being audio webcast and will be archived on our website for approximately 12 months. A telephone replay will be available today by approximately 5:30 p.m. Eastern Time through 5:30 p.m. Eastern Time on November 7. To access the telephone replay, domestic callers should dial 877-660-6853 with conference number 13683414 followed by pound. International callers should dial 201-612-7415 with the same conference number.As a reminder, the information that the presenters discuss today will include forward-looking statements, including statements about Align's future events, product outlook and the expected financial results for the fourth quarter of 2018. These forward-looking statements are only predictions and involve risks and uncertainties that are set forth in more detail in our most recent periodic reports filed with the Securities and Exchange Commission. Actual results may vary significantly and Align expressly assumes no obligation to update any forward-looking statement.We have posted historical financial statements, including the corresponding reconciliations and our third quarter conference call slides on our website under Quarterly Results. Please refer to these files for more detailed information.With that, I'll turn the call over to Align Technology's President and CEO, Joe Hogan. Joe? .Joseph M. Hogan - Align Technology, Inc.Thanks, Shirley. Good afternoon, and thanks for joining us on our call today. I'll provide some highlights on the quarter and then briefly discuss the performance of our two operating segments, clear aligners and scanners. John will provide more detail on our financial results and discuss our outlook for the fourth quarter. Following that, I'll come back and summarize a few key points and open up the call to questions.I'm pleased to report third quarter results with revenue and earnings above our outlook, driven by higher-than-expected Invisalign volume, offset somewhat by lower ASPs and foreign exchange. We also had another record quarter for iTero scanner business. Q3 Invisalign volume increased 5.5% sequentially and up 35.3% year-over-year, reflecting strength across regions and customer channels as well as strong growth from both teen and adult patients.From a product perspective, we saw strength across the Invisalign portfolio, with growth from both the comprehensive and non-comprehensive products, reflecting acceleration in the non-comprehensive category related to expansion of our product portfolio, as well as new sales programs and promotions intended to increase adoption and utilization. We also saw continued strength from international regions, especially Asia-Pacific, which is our second largest region after the Americas in Q3.Finally, during Q3, we trained another record 4,930 doctors, driven by APAC and Latin America, including Invisalign Go doctors, which represent continued expansion of our customer base, especially GP dentists. And Invisalign utilization increased overall to 6.1 cases per doctor, including another record for NA orthodontists.Overall, while I'm pleased with the strong volume growth that we achieved this quarter, there were a couple of unexpected factors that impacted our results. In Q3, Invisalign ASPs were down sequentially due to a combination of promotional programs, unfavorable foreign exchange and product mix shift, partially offset by price increases across all regions. John will provide more details on this in his commentary.Now let's turn to the specifics around our third quarter results and let's start with the results of our America region. For the Americas, Q3 Invisalign case volume was up 5.1% sequentially and 29.1% year-over-year, reflecting better-than-expected volume in both our orthodontist and GP dentist channels, driven by strong teenage case volume from orthodontists and adult patient growth across both customer channels.For the Americas region, year-over-year growth was led by orthos, with another record quarter for North American ortho utilization at 17.4 cases per doctor. Q3 Invisalign utilization for North American GP dentists was up year-over-year at 3.5 cases per doctor, reflecting continued expansion of the GP customer base. We also saw strong volume growth from the dental service organizations or DSOs channel, which increased over 40% year-over-year in Q3.In Q3, we trained over 2,000 new Invisalign doctors in the Americas region, of which 560 were in Latin America alone, primarily Brazil. As the world's second largest market for cosmetic procedures, Brazil is estimated to have more than 1 million new orthodontic case starts each year. In September, we hosted a grand opening event at our new office in São Paulo, Brazil with local doctors to thank them for their ongoing support in helping to build the clear aligner market in Brazil through Invisalign practices.We also hosted the grand opening of our newest facilities in Costa Rica located at San Antonio business center, with plans to open another multimillion-dollar facility located at La Lima later this year. The new facilities will eventually house all of our Costa Rica employees, provide space to accommodate our long-term growth.For international doctors, Q3 was another good quarter with Invisalign case volumes up 6.2% sequentially, reflecting a very strong quarter for APAC, offset by EMEA's seasonality, especially in Southern Europe markets which typically have extended summer holidays. Year-over-year volume growth of 45.6% for international reflects increased utilization and continued expansion of our customer base.",
                 'ALLE': "I would now like to turn the conference over to Mike Wagnes, Vice President, Treasurer and Investor Relations. Please go ahead.Mike Wagnes - Allegion PlcThank you, Andrew. Good morning, everyone. Welcome and thank you for joining us for Allegion's third quarter 2018 earnings call. With me today are Dave Petratis, Chairman, President and Chief Executive Officer; and Patrick Shannon, Senior Vice President and Chief Financial Officer of Allegion. Our earnings release, which was issued earlier this morning and the presentation which we'll refer to in today's call are available on our website at allegion.com. This call will be recorded and archived on our website.Please go to slide number 2. Statements made in today's call that are not historical facts are considered forward-looking statements and are made pursuant to the Safe Harbor provisions of federal securities law. Please see our most recent SEC filings for a description of some of the factors that may cause actual results to materially differ from anticipated results and projections. The company assumes no obligation to update these forward-looking statements.Please go to slide number 3. Our release and today's commentary include non-GAAP financial measures, which exclude the impact of charges related to restructuring, acquisitions, tax reform and debt refinancing in current year and prior year results. We believe these adjustments reflect the underlying performance of the business when discussing operational results and comparing to the prior year periods. Please refer to the reconciliation in the financial tables of our press release for further details.Dave and Patrick will discuss our third quarter 2018 results, which will be followed by a Q&A session. For the Q&A, we would like to ask each caller to limit themselves to one question and one follow-up and then reenter the queue. We will do our best to get to everyone given the time allotted.Please go to slide number 4 and I'll turn the call over to Dave.David D. Petratis - Allegion PlcThanks, Mike. Good morning and thank you for joining us today. Allegion delivered another strong quarter in Q3 and I'm pleased with our operational performance, which has positioned us well to deliver a solid 2018. Allegion saw a strong top-line revenue growth in the third quarter with strength across all regions. The Americas saw solid volume in both the non-residential and residential businesses, as end-market fundamentals continue to be positive, particularly in institutional verticals.Price was very strong in the Americas and remain firmed in Europe. Asia-Pacific saw organic growth rebound nicely in the quarter. Acquisitions continue to contribute to total company revenue growth. Electronics growth globally continues to accelerate and was quite substantial during the quarter, led by the Americas region, which saw nearly 30% growth in electronic products during the quarter. We see the electronics growth outlook continuing to be a long-term positive trend, as more and more products become connected for ease of access.We are driving price realization and productivity actions in an effort to mitigate the substantial inflationary pressures we are experiencing. We did experience margin declines in the quarter as acquisitions were diluted. Excluding the 2018 acquisitions, adjusted operating margins were flat year-over-year. I'm very pleased with our ability to manage those inflationary pressures and maintain organic margins through both pricing and cost actions.In the third quarter, we delivered robust EPS growth, seeing more than 20% expansion in adjusted EPS, driven primarily from operations and tax rate benefits. This EPS performance highlights our continued focus on driving increased shareholder value.Last, we are raising our outlook for revenue and EPS. Total revenue growth is projected to be between 13% and 13.5%, raising from our previous range of 12.5% to 13.5%. Organic revenue is being increased to a range of 5% to 5.5%, up from the 4% to 5% provided last quarter. For adjusted EPS, we are raising the range to $4.43 to $4.50 per share, bringing up the low end of our previous outlook, which was $4.35 to $4.50.Please go to slide 5. In Q3, Allegion delivered strong top-line revenue performance. Revenue from the third quarter with $711.5 million, an increase of 16.8%, inclusive of organic growth of 8.5%. Benefits from acquisition also contributed to the top-line revenue expansion, offsetting the slightly unfavorable currency impact.All regions grew organically. America led the way with organic growth of just over 10% in the quarter, supported by robust growth in electronics, strong pricing, and solid volume in both the nonresidential and residential businesses.The EMEIA regions saw organic growth of 3.4% and Asia-Pacific organic growth rebounded nicely, coming in at nearly 6%. Adjusted operating margins decreased by 140 basis points due to the dilution related to our acquisitions made earlier this year. Excluding the impact of those acquisitions, the base operating margins were flat year-over-year as the business did a good job of driving price realization and productivity savings to combat continued inflationary pressures.",
                 'AGN': "Thank you. I would now like to turn the call over to Daphne Karydas. Please go ahead, ma'am.Daphne Karydas - Allergan PlcThank you, Darla, and good morning, everyone. I'd like to welcome you to the Allergan third quarter 2018 earnings conference call. Earlier this morning we issued a press release reporting Allergan earnings for the quarter ended September 30, 2018. The press release and our slide deck, which we are presenting this morning, are available on our corporate website at www.allergan.com. We are conducting a live webcast of this call, a replay of which will be available on our website after its conclusion. Please note that today's call is copyrighted material of Allergan and cannot be rebroadcast without the company's express written consent.Turning to slide 2, I'd also like to remind you that during the course of this call, management will make projections or other forward-looking remarks regarding future events or the future financial performance of the company. It's important to note that such statements and events are forward-looking statements and reflect our current perspective of the business trends and information as of today's date. Actual results may differ materially from current expectations and projections, depending on a number of factors affecting the Allergan business. These factors are detailed in our periodic public filings with the Securities and Exchange Commission. Allergan disclaims any intent or obligation to update these forward-looking statements except as expressly required by law.All figures discussed during the call refer to non-GAAP. Our GAAP financial metrics and reconciliation from GAAP to non-GAAP metrics can be found in our earnings release issued this morning and posted on our website. In addition, all global and international growth rates referenced this morning are on an ex-FX basis.Turning to slide 3 and our agenda this morning. With us on today's call are Brent Saunders, our Chairman and CEO; Bill Meury, our Chief Commercial Officer; David Nicholson, our Chief R&D Officer; and Matt Walsh, our Chief Financial Officer. Also on the call and available during the Q&A is Bob Bailey, our Chief Legal Officer.With that, I will turn the call over to Brent.Brenton L. Saunders - Allergan PlcThanks, Daphne, and thank you, everyone, for joining our third quarter earnings call this morning. Let's turn to slide 5. As we approach year-end, our business continues to show strong momentum across the board. Our focus on execution is evident in our performance, and we continued to deliver solid results in the third quarter. We remain on track to meet our commitments as we advance our strategy to strengthen our leadership position in our four key therapeutic areas.Our third quarter performance highlights are strategic focus and the durability of our business. Our Medical Aesthetics business delivered another quarter of double-digit growth. As a global leader in the fast-growing market that is expected to double in the next five to seven years, we see exceptional prospects to continue to grow this business. At our Medical Aesthetics Day in September, we provided further insights into the trends and dynamics that are driving the medical aesthetics market and our business.In CNS, VRAYLAR and BOTOX Therapeutic continue to be our anchors, each growing at double-digit rates again in the third quarter. And concerning the richness of our CNS pipeline, the long-term outlook for our CNS business is very good.We continue to demonstrate significant progress in our pipeline with a number of potential key launches over the next three years. This quarter, we sold our medical dermatology assets and acquired Bonti to continue to sharpen our focus. Finally, we continued to generate robust cash flows and maintained a disciplined approach to capital allocation with a strong balance sheet, strategic growth investments and capital returned to our shareholders. We achieved all of this in a quarter where we also experienced some near-term headwinds, some of which were unexpected.During the third quarter, we recalled OZURDEX in certain international markets. David will go over the details, but importantly, the issue that drove the recall was self-identified and no related safety issues have been reported to date. Patient safety is always our number one priority, and we have already implemented corrective actions. We expect to restart the supply of OZURDEX in the impacted markets before the end of this year.Unfavorable currency movements, LOEs and ongoing pressure from payers have been felt industry-wide and we are well positioned to effectively manage through them. Given our strong underlying business momentum in each of our core therapeutic areas and our confidence in driving solid sales and earnings growth, we are raising our outlook for full year 2018.",
                 'ADS': 'Vikki NakhlaThank you, operator. By now, you should have received a copy of the company’s third quarter 2018 earnings release. If you haven’t, please call Advisory Partners at (212) 750-5800. On the call today, we have Ed Heffernan, President and Chief Executive Officer of Alliance Data; and Charles Horn, Chief Financial Officer of Alliance Data.Before we begin, I would like to remind you that some of the comments made on today’s call and some of the responses to your questions may contain forward-looking statements. These statements are subject to the risks and the uncertainties described in the company’s earnings release and other filings with the SEC. Alliance Data has no obligation to update the information presented on the call. Also on today’s call, our speakers will reference certain non-GAAP financial measures, which we believe will provide useful information for investors. Reconciliation of those measures to GAAP will be posted on the Investor Relations website at alliancedata.com.With that, I would like to turn the call over to Ed Heffernan. Ed?Ed HeffernanThanks, Vikki. With me today as always is Charles Horn, our CFO. He’ll update you on the third quarter results and then I will update everyone on our 2018 outlook, talk a little bit about some of the things we have going on and initial thoughts about 2019. We’ll try to get through this and have plenty of time for Q&A. So, Charles?Charles HornThanks, Ed. Pro-forma revenue increased 5% to $2 billion for the third quarter of 2018, led by strong performances at LoyaltyOne and Card Services. EPS increased 28% to $5.39 for the third quarter of 2018, while core EPS increased 17% to $6.26, both aided by lower effective tax rate. That flips in the fourth quarter as we expect a 24% core effective tax rate compared to 14% in the fourth quarter of 2017.Turning to capital deployment, we spent approximately $103 million in share repurchases during the third quarter while dropping our corporate leverage ratio to 2.4x compared to our covenant of 3.5x. Year-to-date we have repurchased slightly over 1 million shares or about 2% of outstanding shares. Cash flow will continue to be used – continue to support share repurchases in our quarterly dividend but will also be used to reduce our corporate leverage ratio to 2.2x or lower by year end. Let’s turn to a recent hot topic which is CECL, the new accounting standard impacting loan loss reserving. The new accounting standard uses the life of the loan methodology to determine expected credit losses. The life of the loan approach is widely viewed as replacing the loss emergence period creating the potential for estimates to cover a longer loss horizon.We are in the early stages for implementation of the standard but the preliminary thought is at that the life of the credit card loan will be less than 12 months of coverage we currently reserve. Key in the determination of the allowance is the ability to provide reasonable and supportable forecast over the life of the loan. Again because we expect the life of credit card loan to be reasonably short, we believe our current forecasting process would cover that period as opposed to those with mortgage or auto loans, which have significantly longer life. In summary, we do not expect the new accounting standard to have any impact to our months coverage.Let’s go to next page and talk about the segments. Starting with LoyaltyOne pro forma revenue increased 8% to $329 million. Breaking down the quarter, AIR MILES’ pro forma revenue decreased 6% from the third quarter of 2017 primarily due to a lower burn rate, that’s the number of miles redeemed and a weaker Canadian dollar. AIR MILES issued increased 3% compared to the third quarter of 2017, benefiting from increased promotional activity by the Bank of Montreal. BrandLoyalty’s revenue increased 29% from the third quarter of 2017 and we expect continued strong growth in the fourth quarter.Turning to Epsilon, revenue decreased 4% to $538 million. Similar to the first half of 2018, combined revenue for our agency and side-based offerings were down double-digits. However, through strong expense management and the favorable shift in revenue mix, we maintained the same level of adjusted EBITDA as of third quarter 2017.Lastly, Card Services revenue increased 10% to $1.16 billion, consistent with the growth in average card receivables, while adjusted EBITDA net increased 4% to $414.3 million. Net loss rates improved sequentially by 50 basis points as the recovery rate improved from high single digits in the first quarter to slightly over 18% in the third quarter. The third quarter reserve rate decreased slightly to 6.5% due to expectations of further improvements in the net principal loss rate in the fourth quarter. The delinquency rate remains slightly elevated primarily due to a slowing growth rate in card receivables, essentially what we call growth map.',
                 'LNT': "I would now like to turn the call over to your host, Susan Gille, Investor Relations Manager at Alliant Energy.Susan GilleGood morning. I would like to thank all of you on the call and the webcast for joining us today. We appreciate your participation. With me here today are Pat Kampling, Chairman and Chief Executive Officer; John Larsen, President; and Robert Durian, Senior Vice President, CFO and Treasurer; as well as other members of the senior management team. Following prepared remarks by Pat, John and Robert, we will take time to take questions from the investment community.We issued a news release last night announcing Alliant Energy's third quarter and year-to-date financial results, updated our 2018 earnings guidance range and announced the 2019 earnings guidance and common stock dividend target. We also provided our annual capital expenditure plan through 2022 and our current estimated total CapEx for 2023 through 2027. This release as well as supplemental slides that will be referenced during today's call are available on the Investor page of our website at www.alliantenergy.com.Before we begin, I need to remind you the remarks we make on this call and our answers to your questions include forward-looking statements. These forward-looking statements are subject to risks that could cause actual results to be materially different. Those risks include, among others, matters discussed in Alliant Energy's press release issued last night and in our filings with the Securities and Exchange Commission. We disclaim any obligation to update these forward-looking statements.In addition, this presentation contains references to non-GAAP financial measures. The reconciliation between non-GAAP and non-GAAP measures are provided in our earnings release and our quarterly report on Form 10-Q, which is available on our website at www.alliantenergy.com.At this point, I'll turn the call over to Pat.Patricia KamplingThanks, Sue. Good morning, and thank you for joining us today. I am pleased to report that we continue to deliver solid financial and operational results. Third quarter did benefit from higher sales through the warmer weather -- the warmer than normal temperatures we enjoyed. On a year-to-date basis, our earnings per share of $1.83 included $0.05 benefit from temperature. Therefore, we are updating our earnings guidance range to $2.13 to $2.19 per share with the new midpoint of $2.16, $0.05 higher than our original guidance for the year. Based on our forecast, 2018 will mark the sixth year in a row that we've achieved at least 5% to 7% earnings per share growth. Robert will provide more details on quarterly results later on the call.Now, let's focus on 2019. The midpoint of our earnings guidance range is $2.24 per share, which is a 6% increase to our forecast of 2018 temperature-normalized non-GAAP earnings per share of $2.11 as shown on Slide 2. This is consistent with our long-term earnings growth objective of 5% to 7% per year through 2022. Additionally, you'll be pleased to know that our Board of Directors has approved a 6% increase of our targeted 2019 annual common stock dividend to $1.42 per share, consistent with our targeted dividend payout ratio of 60% to 70% of earnings.We also issued our 2018 to 2022 capital expenditure plan totaling $7 billion as shown on Slide 3. For your convenience, we provided a walk from the previous capital plan to our current plan on Slide 4.Gas and electric distribution spend did increase in all the years. I would like to point out that the gas distribution CapEx increase of $155 million in 2020 is driven mostly by several anticipated gas expansion projects in Wisconsin. The $205 million decrease in renewables in 2019 was driven mostly by revisions and the timing of our wind expansion spend since we now have finalized our in-service dates for each project. Also, we now assume that the transmission upgrade at the wind farms will be the responsibility of the transmission operators and therefore reduce CapEx for these projects.In the press release, you will notice that we also mentioned that our capital expenditure plan for 2023 through 2027 is currently $5.7 billion, making a 10-year plan total of $12.7 billion. Over 1/2 of our 10-year CapEx plan is for investments in electric and gas distribution systems. These investments will expand automation, standardize voltages and increase underground distribution to make the grid more resilient and improve delivery to customers. We are also working closely with several of our communities to expand our systems in anticipation of economic development opportunities they are pursuing.And speaking of our communities, mother nature was kind to many areas across our service territories during the third quarter. Several of our communities are still being impacted by the devastation caused by tornadoes and major flooding. Marshalltown, Iowa continues the rebuilding process after an EF-3 tornado caused catastrophic damage in July. And in Wisconsin, dozens of homes and businesses suffered major damage during the severe flood that occurred in August and September. This was a challenging summer for many and it has taken an incredible effort by our employees to safely restore service and to generously assist our customers and communities with their time and financial assistance.",
                 'ALL': "I would now like to introduce your host for today's program, Mr. John Griek, Head of Investor Relations. Please go ahead.John Griek - The Allstate Corp.Well, thank you, Jonathan. Good morning, and welcome, everyone, to Allstate's third quarter 2018 earnings conference call. After our prepared remarks, we will have a question-and-answer session.Yesterday, following the close of the market, we filed the 10-Q for the third quarter and posted the news release, investor supplement, and today's presentation on our website at allstateinvestors.com. Our management team is here to provide perspective on these results.As noted on the first slide of the presentation, our discussion will contain some non-GAAP measures, for which there are reconciliations in the news release and investor supplement and forward-looking statements about Allstate's operations. Allstate's results may differ materially from these statements, so please refer to our 10-K for 2017 and other public documents for information on potential risks.Now, I'll turn it over to Tom.Thomas Joseph Wilson - The Allstate Corp.Good morning. Thank you for joining to stay current on Allstate's operating results.Let's begin on slide 2, so the headline year is Allstate's businesses continued to deliver growth and attractive returns. Our strategy is working. We're on pace to achieve our five 2018 operating priorities.At the Property-Liability, year-to-date underlying combined ratio was at the favorable end of the annual outlook range or for nine months for a close to the bottom of the annual outlook, which is good, as you know. The net result was excellent financial results. We also announced a new $3 billion share repurchase program, of which up to $1 billion may be funded with perpetual preferred stock.To go to the table, revenues increased to $10.5 billion, almost $600 million above the prior year quarter. Net income was $833 million in the third quarter of 2018 and adjusted net income per share was $1.93 million, which is 20.6% increase over the prior year quarter. Net income return on equity for the latest 12 months was 17.4% and 15.9% on an adjusted net income basis.Now, let's start on a more macro perspective before we dig into the details. So if you go to slide 3, Allstate's strategy is to grow market share in the personal Property-Liability businesses while expanding our other protection businesses. So, to start with the upper oval, the personal Property-Liability market has four consumer segments and we serve each of these with a differentiated branded product offering. So Allstate agencies provide customers with personal guidance through 38,000 professionals and over 10,000 agencies in nearly every community in America.The Esurance on the other hand provides over 1.6 million policies to customers who prefer to purchase their products online or through a call center. We use this sophisticated analytics across all these businesses to ensure we grow profitability and are at the forefront to using telematics-based offerings. We're building an integrated digital enterprise that uses data analytics technology and process redesign to improve both our effectiveness and our efficiency. For example, we are a leader in using consumer generated – or customer generated photos to quickly settle auto insurance claims.Our strategy also includes expanding other protection businesses by leveraging our brands, customer base, investment expertise, distribution and capital, which are listed in the lower oval and that obviously began in 1957 with life insurance. In 1999, we acquired Allstate Benefits, which provides protection product such as life and disability insurance to employees at the work site. And by leveraging our resources, this business has had a compound annual growth rate of 7% for 18 years.We purchased SquareTrade at the beginning of 2017 and it's growing rapidly and achieving our acquisition objectives. And this year, we also began offering insurance to transportation networks companies such as through Allstate business insurance. And we recently closed the acquisition of InfoArmor, which further expands our portfolio of person identity protection products and services.So, this two part strategy then creates shareholder value in a number of different ways through customer satisfaction, unit growth, attractive returns on capital. It also ensures we have sustainable profitability and a diversified business platform.So, if you turn to slide 4 then as to what we're doing in 2018, the operating priorities we communicate to you and to our employees and teammates. The first three priorities, better serve customers, achieve target economic returns on capital and grow the customer base. They're intertwined to ensure profitable long-term growth.So, customers were better served as the Net Promoter Score improved across all of our major business. Higher customer retention then at Allstate, Esurance and Encompass, as Property-Liability businesses is driving growth.",
                 'GOOG': "Ellen West - Alphabet, Inc.Thank you. Good afternoon everyone and welcome to Alphabet's third quarter 2018 earnings conference call. With us today are Ruth Porat and Sundar Pichai. Now I'll quickly cover the safe harbor. Some of the statements that we make today may be considered forward looking including statements regarding our future investments, our long-term growth and innovation, the expected performance of our businesses and our expected level of capital expenditures.These statements involve a number of risks and uncertainties that could cause actual results to differ materially. For more information, please refer to the risk factors discussed in our Form 10-K for 2017 filed with the SEC. Undue reliance should not be placed on any forward-looking statements and they are made based on assumptions as of today. We undertake no obligation to update them.During this call, we will present both GAAP and non-GAAP financial measures. A reconciliation of GAAP to non-GAAP measures is included in today's earnings press release. As you know we distribute our earnings release through our Investor Relations website located at abc.xyz/investor. This call is also being webcast from our IR website where a replay of the call will be available later today.And now I'll turn the call over to Ruth.Ruth Porat - Alphabet, Inc.Thank you, Ellen. Our revenues in the third quarter continued to benefit from ongoing strength in mobile search with important contributions from YouTube, cloud and desktop search, resulting in consolidated revenues of $33.7 billion, up 21% year-on-year and up 22% in constant currency.For today's call, I will begin with the results for the quarter on a consolidated basis for Alphabet focusing on year-over-year changes. I will then review results for Google followed by Other Bets and will conclude with our outlook. Sundar will then discuss business and product highlights after which we will take your questions.Starting with a summary of Alphabet's consolidated financial performance for the quarter, our total revenues of $33.7 billion reflect a negative currency impact year-over-year of $385 million or $305 million after the impact of our hedging program.Turning to Alphabet revenues by geography, you can see that our performance was strong again in all regions. U.S. revenues were $15.5 billion, up 20% year-over-year. EMEA revenues were $11 billion, up 20% year-over-year. In constant currency terms EMEA grew 19%. APAC revenues were $5.4 billion, up 29% versus last year and up 30% in constant currency. Other Americas revenues were $1.8 billion, up 19% year-over-year and up 28% in constant currency, reflecting weakening of the Brazilian real and the Argentine peso.On a consolidated basis, total cost of revenues including TAC, which I'll discuss in the Google segment results, was $14.3 billion, up 28% year-on-year. Other cost of revenues on a consolidated basis was $7.7 billion, up 36% year-over-year, primarily driven by Google-related expenses. The key drivers were costs associated with our data centers and other operations including depreciation which continue to be affected by a reallocation of certain operating expenses and content acquisition costs, primarily for YouTube.Operating expenses were $11.1 billion, up 26% year-over-year. Once again the biggest increase was in R&D expenses, reflecting our continued investment in technical talent. The growth in sales and marketing expenses reflects increases in sales and marketing head count primarily for cloud and ads followed by advertising investments in cloud, Chromebooks for the back-to-school season and the Google Assistant.G&A expense trends in the third quarter were affected by a number of factors. In particular, the performance fees accrued in connection with recognition of equity security gains which were again partially offset by the reallocation of certain expenses from G&A primarily to other cost of revenues.Stock-based compensation totaled $2.2 billion. Headcount at the end of the quarter was 94,372, up 5,314 from last quarter. Consistent with prior quarters, the majority of new hires were engineers and product managers. In terms of product areas, the most sizable head count increases were in cloud for both technical and sales roles.Operating income was $8.3 billion, up 7% versus last year for an operating margin of 25%. As discussed in the previous two quarters, both operating income and OI&E are affected by the new accounting standard that changes the way companies account for equity security investments. This new standard continues to result in greater volatility. Once again, we've provided a table in our earnings press release to highlight the impact on particular line items.Other income and expense was $1.8 billion which includes $1.4 billion of gains in equity security investments. We provide more detail on the line items within OI&E in our earnings press release.",
                 'MO': "I would now like to turn the call over to Mr. Bill Marshall (00:36), Vice President, Investor Relations for Altria Client Services. Please go ahead, sir.Unverified ParticipantThank you, Laurie. Good morning and thank you for joining us. We're here this morning with Howard Willard, Altria's CEO; and Billy Gifford, Altria's CFO, to discuss Altria's 2018 third quarter and first nine months business results. Earlier today, we issued a press release providing these results, which is available on our website at altria.com and through the Altria Investor app.During our call today, unless otherwise stated, we're comparing results to the same period in 2017. Our remarks contain forward-looking and cautionary statements, and projections of future results. Please review the forward-looking and cautionary statement section at the end of today's earnings release for various factors that could cause actual results to differ materially from projections. Future dividend payments and share repurchases remain subject to the discretion of Altria's board. The timing of share repurchases depends on marketplace conditions and other factors. Altria reports its financial results in accordance with U.S. generally accepted accounting principles.Today's call will contain various operating results on both a reported and adjusted basis. Adjusted results exclude special items that affect the comparability of reported results. Descriptions of these non-GAAP financial measures and reconciliations are included in today's earnings release.With that, I'll turn the call over to Howard.Howard A. Willard - Altria Group, Inc.Thanks, Bill, and good morning, everyone. Altria delivered excellent third quarter adjusted diluted earnings per share growth of 20% and continued to return large amounts of cash to our shareholders. Our tobacco businesses are successfully executing against their strategies, while making strategic investments to drive long-term success. Before moving to our third quarter results, I'd like to address recent FDA activity.In September, the FDA asked several companies, including Altria, to provide plans to address underage use of e-vapor products. We welcomed FDA's action and we agreed that the reported rise in underage use of e-vapor products is alarming and immediate action should be taken. We're also concerned that use of e-vapor products may jeopardize the harm reduction opportunity for e-vapor. We recently met with Commissioner Gottlieb to discuss steps that could be taken to address underage access and use.Consistent with our discussion with the FDA and because we believe in the long-term promise of e-vapor products and harm reduction, we're taking immediate action to address this complex situation. First, Nu Mark will remove from the market MarkTen Elite and Apex by MarkTen pod-based products until these products receive a market order from the FDA or the youth issue is otherwise addressed.Second, for our remaining MarkTen and Green Smoke cig-a-like products, Nu Mark will sell only tobacco, menthol and mint varieties. Nu Mark will discontinue the sale of all other flavor variants of our cig-a-like products until these products receive a market order from the FDA or the youth issue is otherwise addressed. Although we don't believe we have a current issue with youth access or use of our e-vapor products, we are taking this action because we don't want to risk contributing to the issue.Additionally, we will support federal legislation to establish 21 as the minimum age to purchase any tobacco product. We think it makes sense to accomplish this through a phased-in approach. For context, we estimate that approximately 5% of adult tobacco consumers are legal age through 20, and that this age demographic represents approximately 2% of cigarette industry volumes, 4% of smokeless industry volumes and 15% of e-vapor industry volumes.We, of course, recognize the impacts these decisions will have on our consumers, trade partners, suppliers and others. We believe these actions are essential to addressing the youth e-vapor epidemic and preserving the long-term harm reduction opportunity for e-vapor products. We support adult tobacco consumer choice and the promise of tobacco harm reduction, and we fully intend to operate compelling portfolio of e-vapor products for adult smokers and vapers.Through the FDA's product review pathways all but (05:08) underage use of e-vapor is addressed. After removing Nu Mark's pod-based products and cig-a-like flavor variants, approximately 80% of Nu Mark's e-vapor volume in the third quarter of 2018 will remain on the market. These actions are outlined in our written response to the FDA, which was posted earlier this morning to altria.com.With that, let's move now to our operating results. The smokeable products segment performed in line with our expectations, while adjusted operating companies income in the third quarter was essentially flat from the prior year, PM USA continues to make progress stabilizing Marlboro share through investments in the brand's equity. Marlboro's retail share decreased 0.1 of a share point in the third quarter to 43.1%, but is unchanged from its fourth quarter 2017 share. Billy will provide additional detail on our brand equity investments in a minute.",
                 'AMZN': "For opening remarks, I will be turning the call over to the Director of Investor Relations, Dave Fildes. Please go ahead.Dave Fildes - Amazon.com, Inc.Hello, and welcome to our Q3 2018 financial results conference call. Joining us today to answer your questions is Brian Olsavsky, our CFO. As you listen to today's conference call, we encourage you to have our press release in front of you, which includes our financial results as well as metrics and commentary on the quarter. Please note, unless otherwise stated, all comparisons in this call will be against our results for the comparable period of 2017.Our comments and responses to your questions reflect management's views as of today, October 25, 2018 only, and will include forward-looking statements. Actual results may differ materially. Additional information about factors that could potentially impact our financial results is included in today's press release and our filings with the SEC, including our most recent annual report on Form 10-K and subsequent filings.During this call, we may discuss certain non-GAAP financial measures. In our press release, slides accompanying this webcast and our filings with the SEC, each of which is posted on our IR website, you will find additional disclosures regarding these non-GAAP measures, including reconciliations of these measures with comparable GAAP measures.Our guidance incorporates the order trends that we've seen to date and what we believe today to be appropriate assumptions. Our results are inherently unpredictable and maybe materially affected by many factors, including fluctuations in foreign exchange rates, changes in global economic conditions and customer spending, world events, the rate of growth of the Internet, online commerce and cloud services, and the various factors detailed in our filings with the SEC.Our guidance also assumes, among other things, that we don't conclude any additional business acquisitions, investments, restructurings or legal settlements. It's not possible to accurately predict demand for our goods and services, and therefore, our actual results could differ materially from our guidance.With that, we'll move to Q&A. Operator, please remind our listeners how to initiate a question.Question-and-Answer SessionOperatorAt this time, we will now open the call up for questions. Thank you. Our first question comes from the line of Justin Post with Merrill Lynch. Please proceed.Justin Post - Bank of America Merrill LynchGreat. Thank you for taking my question. I guess the big one is the deceleration in unit growth or online stores, which are probably related to that. I know it's a tough 3Q comp, but could you comment a little bit about that? And then kind of what initiatives could be most interesting to maybe reaccelerate that over the next couple of years? What categories? Thank you.Brian T. Olsavsky - Amazon.com, Inc.Thank you, Justin. Yeah, let me just remind you a couple of things from last year. We had two reactions on our Super Saver Shipping threshold in the first half of the year between February and May. That did spur a lot of unit growth in the second and third quarter. We also have issue with digital content, not an issue, but the fact that digital content is moving to subscriptions, Amazon Music Unlimited and Kindle Unlimited in particular. It's been really popular.I'll just remind you that the units – those do not count in our unit totals nor do the units from Whole Foods Market. So, yeah, I would say essentially with that backdrop, we're still very, very encouraged by the demand and the reception from customers on the consumer side. We have Amazon fulfilled units are still growing faster than paid units. 3P is now up to 53% of total paid units.In-stock is very strong, especially as we head into the holiday period. I think we're well positioned for the holiday. We have over 100 million Prime eligible items that are available for FREE Two-Day Shipping for Prime members. And again, when we're talking about the unit deceleration, a lot of the fastest growing areas, things like subscription services, AWS and advertising are not caught in that metric.Dave Fildes - Amazon.com, Inc.And, Justin, this is Dave. Just to add on to that. You mentioned the online stores. Just a reminder, there is a little bit of impact from the revenue recognition. So you see the online stores revenue growing about 11% ex-FX, to be higher than that, but for the adoption ad standard. So there's a little bit of a headwind there as well.OperatorThank you. Our next question comes from line of Mark Mahaney with RBC Capital Markets. Please proceed.",
                 'AEE': "It is now my pleasure to introduce your host, Mr. Andrew Kirk, Director of Investor Relations for Ameren Corporation. Thank you, Mr. Kirk. You may begin.Andrew KirkThank you, and good morning. On the call with me today are Warner Baxter, our Chairman, President and Chief Executive Officer; and Marty Lyons, our Executive Vice President and Chief Financial Officer; as well as other members of the Ameren management team. Warner and Marty will discuss our earnings results and guidance as well as provide a business update. Then we will open the call for questions. Before we begin, let me cover a few administrative details. This call contains time-sensitive data that's accurate only as of the date of today's live broadcast, and redistribution of this broadcast is prohibited.To assist with our call this morning, we have posted a presentation on the amereninvestors.com Web site home page that will be referenced by our speakers. As noted on page two of the presentation, comments made during this conference call may contain statements that are commonly referred to as Forward-Looking Statements. Such statements include those about future expectations, beliefs, plans, strategies, objectives, events, conditions and financial performance. We caution you that various factors could cause actual results to differ materially from those anticipated. For additional information concerning these factors, please read the Forward-Looking Statement section in the news release we issued today and the Forward-Looking Statements and Risk Factors sections in our filings with the SEC.Lastly, all our per share earnings amounts discussed today during today's presentation including earnings guidance are presented on a diluted basis unless otherwise noted.Now here's Warner, who will start on page four of the presentation.Warner BaxterThanks, Andrew. Good morning everyone, and thank you for joining us. Earlier today, we announced third quarter 2018 core earnings of $1.50 per share compared to core earnings of $1.24 per share in the third quarter of 2017. Third quarter 2018 results exclude non-cash non-core charge of $0.05 per share related to federal income tax reform.The year-over-year increase was driven by higher retail sales, primarily due to warmer summer temperatures, as well as earnings on increased infrastructure investments. The comparison also benefited from timing differences related to federal income tax reform, which we do not expect will impact full-year results. Marty will discuss these and other factors driving the quarterly results in more detail in a moment.I am also pleased to report that we continue to successfully execute our strategic plan across all of our businesses, which I will touch on in more detail in a few moments.That fact [ph], and coupled with strong retail sales primarily due to the warm summer weather enabled us to raise our 2018 earnings guidance for the second time this year. Our 2018 core earnings guidance range is now $3.35 per share to $3.45 per share, up from our prior GAAP and core guidance range of $3.15 per share to $3.35 per share.Moving to page five, here we reiterate our strategic plan which we have been executing very well throughout 2018, and over the last several years. That plans is expected to continue to result in strong long-term investments and earnings growth. The first pillar of our strategy stresses investing in and operating our utilities in a manner consistent with existing regulatory frameworks. This strategy has driven our multiyear focus on investing in energy infrastructure for the long-term benefit of customers in jurisdictions that are supported by modern, constructive regulatory frameworks.Today, I am pleased to say that we are allocating meaningful amounts of capital to each of our business segments as all are now operating under constructive frameworks. On September 1st, Ameren Missouri began using plant and service accounting enabled by Senate Bill 564. This legislation improves our ability to earn a fair return on Ameren Missouri capital investments, and will drive significant incremental investments in energy infrastructure in the future. I'll say more about Senate Bill 564 in a moment. But it is safe to say that we are excited to bring many of the same infrastructure investment benefits to Missouri that our Illinois customers enjoy today.Speaking about our Ameren Illinois Electric and Natural Gas distribution operations, we invested approximately $625 million in Illinois electric and natural gas delivery infrastructure projects in the first nine months of this year, including those that are part of Ameren Illinois' modernization action plans. The electric projects, enabled by the Illinois Energy Infrastructure Modernization Act, have exceeded the job creation goals, and we are on track to meet or exceed our investment, reliability, and advance metering goals. Ameren Illinois customers are experiencing fewer and shorter power outages as a result of electric grid upgrades.Since the modernization program began in 2012, the installation of storm-resilient utility poles, automated switches, and an upgraded distribution grid have resulted in a 19% reduction in annual electricity service interruptions on average. And when customers do experience an outage, Ameren Illinois is restoring power 17% faster on average than prior to when the program began. Further, installations of advanced electric meters and gas meter modules were either on or ahead of schedule. Through the end of September, Ameren Illinois has installed about 985,000 smart electric meters, and 510,000 gas meter modules that provide customers with enhanced energy usage data and access to programs to help them save on their energy bills.",
                 'AAL': "And now, I would like to turn the conference over to your moderator, Managing Director of Investor Relations, Mr. Dan Cravens.Daniel E. Cravens - American Airlines Group, Inc.Thanks, and good morning, everyone. And welcome to the American Airlines Group third quarter 2018 earnings conference call.Joining us on the call this morning is Doug Parker, Chairman and CEO; Robert Isom, President; and Derek Kerr, our Chief Finance Officer. Also in the room for a question-and-answer session are several of our senior executives, including Maya Leibman, Chief Information Officer; Steve Johnson, our EVP of Corporate Affairs; Elise Eberwein: our EVP of People & Communications; and Don Casey, our Senior Vice President of Revenue Management.Like we normally do, Doug will start the call with an overview of our financial results. Derek will then walk us through the details on the third quarter and provide some additional information on guidance for the fourth quarter. Robert will then follow with commentary on the operational performance and revenue environment. And then, after we hear from those comments, we'll open the call for analysts' questions, and lastly questions from the media. To get in as many questions as possible, please limit yourself to one question and a follow-up.Before we begin, we must state that today's call does contain forward-looking statements including statements concerning future revenues and cost, forecast of capacity, traffic, load factor, fleet plans and fuel prices. These statements represent our predictions and expectations as to future events, but there are numerous risks and uncertainties that could cause actual results to differ from those projected. Information about some of these risks and uncertainties can be found in our earnings press release issued this morning and our Form 10-Q for the quarter ended September 30, 2018 that was also issued this morning.In addition, we will be discussing certain non-GAAP financial measures this morning such as pre-tax profit and CASM, excluding unusual items. A reconciliation of those numbers to the GAAP financial measures is included in the earnings release and that can be found on our website. A webcast of this call will be archived on the website as well, and the information that we're giving you on the call is as of today's date and we undertake no obligation to update the information subsequently.Thanks again for joining us. And at this point, I'll turn the call over to our Chairman and CEO, Doug Parker.William Douglas Parker - American Airlines Group, Inc.Thanks, Dan. Thanks everyone for joining us. Today, we reported third quarter 2018 pre-tax profit of $688 million, excluding net special items. Those results include our highest ever revenue performance, thanks to our 130,000 hard-working team members. But unfortunately, a rising fuel prices outpaced that increase in revenues. Higher jet fuel prices alone increased our quarterly expenses by over $750 million versus the same quarter last year. And therefore, our pre-tax earnings, excluding specials for the quarter, were $485 million lower than the third quarter 2017.The declining earnings has been met with a declining stock price, which neither we nor our investors are happy about. The good news is, we're extremely bullish on the future of American and for good reason. This disconnect between the stock price and our view of the future seems to us like a buying opportunity, and we're happy to be here to talk to you all about it.So, look, there are five reasons that we're so bullish. First, we have extensive revenue initiatives underway that are expected to bring more than $1 billion in revenue improvements to American in 2019 versus 2018. Importantly, the drivers of this value are not share shift because of a better product like new airplanes or industry-leading Wi-Fi or world-class clubs and lounges, though we certainly believe some upside exist in that regard. This is a value that'll happen as we simply execute against known projects such as project segmentation, fleet reconfiguration and international network restructuring.Second, and we also expect about $300 million in cost improvements in 2019 versus this year. That's the result of our One Airline project, which has been expanded and accelerated in light of higher fuel costs and Derek will discuss that further.Third, we have the opportunity to grow where we have a real competitive advantage. We have, what we believe, will be the lowest growth plans in the industry for 2019, but we also have what we believe are the best growth prospects. We have 15 gates opening at our largest and most profitable hub in Dallas/Fort Worth in early 2019. We have routes in and out of Dallas/Fort Worth that will immediately generate higher than average profitability versus the marginal profitability that airline growth usually generates.Fourth, we're dedicated to improving our operating reliability. And we've been steadily improving the operating reliability of American according to plan in each year since our merger in late 2013. But that trend changed in the summer of 2018, and we backed (00:05:04) a little bit. As Robert will discuss, we've rededicated ourselves to producing the best operational reliability since our merger in 2019 and that's our top corporate priority. The work has already begun showing some great results, so this is an even more upside for 2019.And then fifth, we are nearing the end of our major post-merger capital expenditure requirement. Our capital expenditures in American have averaged $5.3 billion per year in the five years since the merger. That over $25 billion is by far the most any carrier has invested in its fleet, product and team in the history of commercial aviation. And the result is a valuable set of assets that'll serve our shareholders well for decades to come.And we're going to spend a little under $5 billion in 2019 as we have one more aircraft order to fund. But after that, we're largely done with the backlog. And our CapEx drops precipitously to approximately $3 billion in 2020, $2 billion in 2021 and we expect it will remain in the $2 billion to $3 billion range thereafter.So, because of all of those items, we're excited about our near and long-term future. We're confident that American will return to revenue outperformance and earnings growth in 2019 and beyond. Now, it sounds like we're extremely optimistic because we are, but please don't mistake confidence for indifference. We're extremely focused on results and execution and completing the hard work necessary to deliver this value. We've just happen to be confident it will happen because we know we have the right plan in place and the right people to deliver it. We look forward to proving that over time.And with that, I'll turn it over to Derek and Robert.",
                 'AEP': "I would now like to turn the conference over to Bette Jo Rozsa. Please go ahead.Bette Jo Rozsa - American Electric Power Co., Inc.Thank you, Cynthia. Good morning, everyone, and welcome to the third quarter 2018 earnings call for American Electric Power. Thank you for taking the time to join us today. Our earnings release, presentation slides, and related financial information are available on our website at aep.com.Today, we will be making forward-looking statements during the call. There are many factors that may cause future results to differ materially from these statements. Please refer to our SEC filings for a discussion of these factors. Our presentation also includes references to non-GAAP financial information. Please refer to the reconciliation of the applicable GAAP measures provided in the appendix of today's presentation.Joining me this morning for opening remarks are Nick Akins, our Chairman, President and Chief Executive Officer; and Brian Tierney, our Chief Financial Officer. We will take your questions following their remarks.I will now turn the call over to Nick.Nicholas K. Akins - American Electric Power Co., Inc.Okay. Thanks, Bette Jo. Good morning, everyone, and welcome again to AEP's third quarter 2018 earnings call. We just completed another financially strong quarter given positive weather results, continued economic growth, albeit moderated in most sectors of the economy, and continued resolution of regulatory related matters. The headlines for this quarter are not only that our board decided to raise the quarterly dividend by 8.1% to $0.67 a share earlier this week, we're also adjusting our 2018 guidance range upward from $3.75 to $3.95 per share to $3.88 to $3.98 per share.The economy in our service territory continues to grow mainly in oil and gas-related industries, with others such as chemical industries, primary metals, et cetera, are tempering because of tariffs and a strengthening U.S. dollar. Brian will cover that in more detail, but overall, AEP continues to deliver on its commitment of providing steady, consistent dividend growth commensurate with the long-term earnings growth expectation of 5% to 7%.As for the specifics for the quarter and the year-to-date, GAAP and operating earnings for third quarter 2018 came in at $1.17 per share and $1.26 per share, respectively, versus third quarter 2017 GAAP and operating earnings of $1.11 and $1.10 per share, respectively. This brings 2018 year-to-date GAAP and operating earnings to $3.17 per share and $3.23 per share, respectively, versus 2017 GAAP, and operating earnings of $3.07 per share and $2.82 per share, respectively.The difference between GAAP and operating for the quarter and for year-to-date 2018 are primarily due to an impairment taken related to the Racine Hydroelectric Plant, severance charges taken in response to announced plant closures, and economic hedging activities. This has been another very positive quarter financially and operationally that should bode well for ending the year in a positive fashion.Because of our belief that we are and continue to be on a firm 5% to 7% earnings trajectory buoyed by a strong base plan into the future, our board was very comfortable increasing the dividend by 8.1%, well within our 60% to 70% targeted payout ratio.Moving to the regulatory activity, at this point, the notable rate case activity really includes Oklahoma and APCo West Virginia. In late September, we filed another base case in Oklahoma at PSO, our third try to correct the chronic under earning situation in Oklahoma. This case has a requested net increase of $68 million with an ROE of 10.3%.We have proposed a performance-based rate mechanism that adjusts for certain customer satisfaction and quality of service metrics with the intent of reducing the regulatory lag while ensuring a positive customer experience. The proposed procedural schedule has now been set that provides for interim rates subject to refund going into effect in April of 2019. We are certainly hoping for a better outcome based upon the operational performance of PSO over the years, a financially healthy PSO would be well deserved.Regarding the West Virginia case that was filed back in early May, we received staff and other intervener testimony that while disappointing regarding adjustments such as a lower ROE, exclusion of certain known reasonable expenses after the test year and lower depreciation rates, we still believe there is an opportunity to enter into constructive settlement discussions to achieve a more reasonable outcome. Procedural schedules have been set in this case with an expected order in late February, with rates going into effect in March 2019.So now, I'll move to the equalizer chart. I'm going to hold off on a discussion of the premier regulated energy company until a little bit later. But as we look at the equalizer chart, I call it that because it sort of looks like it with the balls of different sizes of the companies, overall, we're seeing a 10.1% ROE. We generally project the ROE for our regulated segments combined to be at or near the 10% range.",
                 'AXP': "And I’ll now turn the meeting over to our host, Head of Investor Relations, Mr. Edmund Reese. Please go ahead, sir.Edmund ReeseThank you, Lori. Welcome. We appreciate all of you joining us for today's call. The discussion contains certain forward-looking statements about the company's future financial performance and business prospects, which are based on management's current expectations and are subject to risks and uncertainties. Factors that could cause actual results to differ materially from these forward-looking statements are set forth within today's presentation slides and in the company's reports on file with the Securities and Exchange Commission.The discussion today also contains certain non-GAAP financial measures. Information relating to comparable GAAP financial measures may be found in the third quarter 2018 earnings release and presentation slides, as well as the earnings materials from prior periods that may be discussed, all of which are posted on our website at ir.americanexpress.com. We encourage you to review that information in conjunction with today's discussion.Today's discussion will begin with Steve Squeri, Chairman and CEO, who will start the call with some remarks about the company's progress and results; and then, Jeff Campbell, Chief Financial Officer, will provide a more detailed review of Q3 financial performance. Once Jeff completes his remarks, we will move to a Q&A session on the quarter's results, with both Steve and Jeff.With that, let me turn it over to Steve.Steve SqueriThanks, Edmund and good afternoon, everyone. I'll start with some of the key highlights from the third quarter results we released earlier this afternoon. From there, I'll give an overview of the year-to-date results. And before turning it over to Jeff for a more detailed discussion of our financial performance, I’ll provide a quick progress report on the four strategic imperatives that we focused the organization on at the beginning of the year.In the third quarter, we saw a continued momentum in our business that was in line with the strong growth of the first half. Adjusted revenues grew 10% and we delivered EPS of $1.88. I feel really good about the company's performance. We’ve had several quarters of high revenue growth and in fact this marks the sixth consecutive quarter with adjusted revenue growth of at least 8%. Our growth has been broad based, driven by a well balanced mix of card member spending, fees and loans and spread across geographies, businesses and customer segments.We have continued to control our operating expenses and that provides the flexibility to make investments in our brand, customer benefits and digital capabilities. As I reflect on the first nine months of the year, I see very consistent and positive trends. We generated a healthy level of top line revenue growth and delivered strong bottom line results each quarter. Year-to-date, our revenue growth is 10% and our earnings per share growth is slightly higher than that, after adjusting for the tax act of 2017.We're gaining spending and lending share in almost all the countries in which we operate. At the same time, our customer engagement expenses, which are composed of rewards, card member services and marketing are growing a little faster than revenue. We are seeing very good payback from the targeted enhancements we've made to our customer value propositions. But that does translate into some margin pressure.The remainder of our cost on operating expenses. For us, these are primarily scalable infrastructure costs and they are growing at a much slower rate than revenues. So in effect, the margin compression created by higher customer engagement expenses is being partially offset by OpEx leverage. And as many of you know, we have a proven track record of disciplined control on operating expenses, while growing the rest of the business. We're delivering strong results in a highly competitive and regulated environment where there are higher costs associated with driving growth. And going forward, we are focused on sustaining the high levels of revenue growth that have been delivering steady and consistent double digit EPS growth. We believe that the best way to do that is to invest in share, scale and relevance for the long term.Turning back to our year-to-date performance, we feel good about the advantages that come from our integrated business model and the progress we're making on our four strategic imperatives, which I'll take you through now. We continue to expand our leadership in the premium consumer space. We introduced the new Gold Card in the US with an enhanced value proposition and an innovative set of rewards. We continued to refresh our international product line with enhancements to our Platinum Card in Australia, Singapore in Japan. The US Platinum Card continued its strong performance. New consumer accounts are up more than 50% and about half of those are for millennials. Building on our strong position in commercial payments, small and medium sized businesses continue to be our fastest growing customer segment worldwide, with particular strength in our international regions. Continuing to strengthen our global integrated network, we continue to add more places worldwide, where American Express cards are accepted.While I could expand much more in our progress in these first three areas, today, I want to focus more on our fourth priority, making American Express a more essential part of our customers’ digital lives. Given the announcement we made earlier today about an expanded relationship with PayPal, I wanted to put the many things we are doing here in context. Over the past year, we've steadily gained traction on this critical imperative that cuts across every part of our integrated business model.",
                 'AIG': 'Good day and welcome to the AIG’s Third Quarter 2018 Financial Results Conference Call. Today’s conference is being recorded.At this time, I would like to turn the conference over to Ms. Liz Werner, Head of Investor Relations. Please go ahead, ma’am.Liz WernerThank you, April.AIG is not under any obligation and expressly disclaims any obligation to update any forward-looking statements, whether as a result of new information, future events or otherwise. Today’s presentation may contain non-GAAP financial measures. The reconciliation of such measures to the most comparable GAAP figures is included in the slides for today’s presentation and our financial supplement, both of which are available on our website. As a reminder, for this morning’s call, our Q&A session will have one question with one follow-up. Please get back into queue, if you would like to ask additional questions.On this morning’s call, you’ll hear from our senior management team including CEO, Brian Duperreault; CFO, Sid Sankaran; CEO of GI, Peter Zaffino; GI’s Chief Actuary, Mark Lyons; and CEO of L&R, Kevin Hogan.At this time, I’d like to turn the call over to Brian.Brian DuperreaultThank you, Liz, and good morning, everyone.Our third quarter results reflected volatility due to 14 global catastrophes, particularly the Japan typhoons. We continue to execute on our reinsurance strategy, one of our key initiatives, which I will cover in a more detail and which Peter will also review in his remarks. Other actions we’re taking to improve our underwriting capabilities and profitability are taking hold, and we started to see some of the resulting benefits in our third quarter results.We continue to expect to deliver an underwriting profit including AAL for General Insurance as we exit 2018. Over the course of the last year, Peter and his team have made significant progress in executing on our reinsurance strategy. And this work will continue as we approach the January 1 renewal season. So far this year, we’ve lowered our North American CAT cover attachment point from a per occurrence of $1.5 billion to an aggregate of $750 million and added an additional international cover.As you will hear from Peter in his remarks, AIG’s national market share in Japan is 6%, and in the area most impacted, it’s 10% on the average. While our Japan reinsurance program was renewed in January 2018 maintaining its historical structure which included two separate towers for the commercial and personal insurance business, we have been working diligently throughout this year to get a single structure in place for 2019 to reduce our net exposure on both a frequency and severity basis. We are pleased with the contributions and balance that Validus brings to our business mix. The disciplined underwriting and risk approach that Validus takes was most evident this quarter in the estimated net CAT loss of approximately $200 million, which was in line with peers. Validus was neutral to our accident year results quarter this quarter and remains on track to contribute approximately 1 point to a combined ratio improvement as we exit 2018.Our recent announcement of the pending acquisition of Glatfelter will provide further balance to General Insurance by improving our position in the programs market with the addition of one of the most respected firms in this space to our portfolio of businesses. The closing of this transaction is expected to occur next week.Turning to reserves, we welcome Mark Lyons, to the team this summer, and he has hit the ground running in reviewing our actuarial processes and procedures. Net reserve additions of $170 million in the third quarter reflect the work Mark and his team performed relating to approximately 75% of our book. Year-to-date, net reserves development was flat.Peter will provide an additional detail on General Insurance in his prepared remarks. And Mark is joining the call and will give you more color on the work he has done on General Insurance reserves.L&R delivered another solid quarter, notwithstanding challenging year-over-year comparisons that reflect the impact of annual assumption reviews. Underlying ROE continued in double digits, and in particular our investments in businesses over the last few years are beginning to bear fruit with strong growth in Individual Retirement and life in particular. Sid will discuss the results of the third quarter actuarial review and Kevin will elaborate on the performance of this well-positioned business, which serves some of the most -- the world’s most important needs, the need for sources of savings, lifetime income, and protection. Our steps to reduce exposures on our Legacy book and to allocate capital more efficiently underscore our capital management discipline and focus on long-term shareholder value. The sale of 19.9% of Fortitude Re, formerly known as DSA Re, to the Carlyle Group is imminent and will free up capital as well as provide a platform for potential growth.',
                 'AMT': "I would now like to turn the conference over to our host, Senior Director of Investor Relations, Mr. Igor Khislavsky. Please go ahead.Igor Khislavsky - American Tower Corp.Good morning and thank you for joining American Tower's third quarter 2018 earnings conference call. We've posted a presentation, which we will refer to throughout our prepared remarks, under the Investor Relations tab of our website, www.americantower.com. Our agenda for this morning's call will be as follows.First, I'll provide a few highlights from our financial results. Next, Jim Taiclet, our Chairman, President, and CEO, will provide some brief commentary focusing on key technology trends in the U.S. And finally, Tom Bartlett, our Executive Vice President and CFO, will provide a more detailed review of our third quarter results and updated full year outlook. After these comments, we'll open up the call for your questions.Before I begin I'll remind you that this call will contain forward-looking statements that involve a number of risks and uncertainties. Examples of these statements include, our expectations regarding future growth, including our 2018 outlook; capital allocation and future operating performance; the pacing and magnitude of Indian carrier consolidation and its impacts on American Tower; assumptions around our pending and recently closed acquisitions and other transactions; and any other statements regarding matters that are not historical facts.You should be aware that certain factors may affect us in the future and could cause actual results to differ materially from those expressed in these forward-looking statements. Such factors include the risk factors set forth in this morning's earnings press release, those set forth in our Form 10-K for the year ended December 31, 2017 as updated in our Form 10-Q for the quarter ended June 30, 2018, and in the other filings we make with the SEC. We urge you to consider these factors and remind you that we undertake no obligation to update the information contained in this call to reflect subsequent events or circumstances.Now please turn to slide 4 of our presentation, which highlights our financial results for the third quarter of 2018. During the quarter, our property revenue grew 5.8% to $1.75 billion. Our adjusted EBITDA grew 5.3% to nearly $1.1 billion. Our consolidated adjusted funds from operations grew about 10% to $821 million, and consolidated AFFO per share increased by nearly 7% to $1.85. Finally, net income attributable to American Tower Corporation common stockholders increased by about 23% to $367 million, or $0.83 per diluted common share.And with that, I'll turn the call over to Jim.James D. Taiclet - American Tower Corp.Thanks, Igor, and good morning to everyone on the call.The highlight of our third quarter was our U.S. property segment organic tenant billings growth of 7.4%, leading us to raise our 2018 expectations for that metric to above 7% for the full year. Unlimited data plans and increasing mobile video consumption continue to drive additional spectrum deployments and equipment installations by our domestic tenants to support 4G network technology, and that's leading to those elevated growth rates.Moreover, our major U.S. customers are beginning to embark on tangible plans for 5G technology, which provides a relevant backdrop to our usual third quarter topic of technology development. But before getting into the details of those trends, I'd like to spend a few minutes on our comprehensive agreement with Tata, which I'm pleased to disclose in this morning's press release.We've been working amicably with the Tata Group for nearly a year to reach an agreement that satisfies three main objectives for us while also respecting our partner's goals of an orderly exit from both the mobile business and from our tower joint venture. Those three objectives for ATC which we believe we've attained through this agreement, were to: first, preserve our ability to achieve American Tower's long-term return on investment targets in India by securing economic value for Tata Teleservices leases to be terminated while at the same time retaining a portion of the run rate through new leases with other Tata Group businesses; second, to better position ATC India for growth in the post-consolidation phase of the Indian mobile market in the 2020 timeframe and beyond; and thirdly, to set the timeline to evolve the joint venture through the replacement of Tata as a partner with either increased ATC ownership, an additional partner, or some combination.Tom will provide additional details of the agreement during his remarks, so now let's jump into our regularly scheduled technology discussion. My remarks are focused today on technology trends in the U.S., which remains our largest market, generating the bulk of our company's cash flows. We also expect that over time, these trends will follow across our international markets.As I noted earlier, the overwhelming driver of tower demand today is 4G technology, and we believe that will remain the case well into the 2020s. However, as momentum for 5G builds, a number of trends in network deployments are expected to increasingly contribute to our demand profile as well.",
                 'AWK': 'Following the earnings conference call, an audio archive of the call will be available through November 8, 2018. U.S. callers may access the audio archive toll-free by dialing 1-877-344-7529. International callers may listen by dialing 1-412-317-0088. The access code for replay is 10125306. The audio webcast will be available on American Water’s Investor Relations homepage at ir.amwater.com through December 1, 2018.I would now like to introduce your host for today’s call, Ed Vallejo, Vice President of Investor Relations. Mr. Vallejo, you may begin.Ed VallejoThank you, Karl. Good morning, everyone, and thank you for joining us for today’s call. We will keep the call to about an hour and at the end of our prepared remarks, we will open the call up for your questions.During the course of this conference call, both in our prepared remarks and to address your questions, we may make forward-looking statements that represent our expectations regarding our future performance or other future events. These statements are predictions based upon our current expectations, estimates and assumptions. However, since these statements deal with future events, they are subject to numerous known and unknown risks, uncertainties and other factors that may cause actual results to be materially different from the results indicated or implied by such statements. Additional information regarding these risks, uncertainties and factors as well as a more detailed analysis of our financials and other important information is provided in the earnings release and in our Form 10-Q as filed with the SEC.Reconciliations for non-GAAP financial information discussed on this conference call, including adjusted earnings per share both as historical financial information and as earnings guidance and our adjusted regulated O&M efficiency ratio, can be found in our earnings release and in the appendix of the slide deck for this call. Also, this slide deck has been posted to our Investor Relations page of our website and will remain available through December 1, 2018. All statements made during this call related to earnings and earnings per share refer to diluted earnings and earnings per share.And with that, I will now turn the call over to American Water’s President and CEO, Susan Story.Susan StoryThanks, Ed. Good morning everyone and thanks for joining us. As always, after my opening remarks, you will hear from our COO, Walter Lynch on regulated business highlights. And then our CFO, Linda Sullivan on third quarter financial results. However, Linda will join me in a few minutes as we talk a bit more about our Keystone operations before turning it over to Walter.We’re pleased to report another strong quarter of performance. Our third quarter adjusted earnings per share were up 11.1% compared to 2017 and our nine months year-to-date grew 11.5% year-over-year. The foundation for our growth continues to be the capital investment we make in regulated operations. We invested a total of $1.5 billion during the first nine months of the year with $1.1 billion for regulated infrastructure and $383 million for acquisitions, which includes our acquisition of Pivotal.We had several positive events this quarter, which Walter will discuss more fully in just a few minutes. To mention just a few, we received a BPU approved settlement in our New Jersey rate case and reached a stipulated agreement with a consumer advocate and PSC staff in West Virginia. We received a unanimous five to zero vote by the California Public Utilities Commission for our Monterey Peninsula Water Supply Project, which includes both water reuse capabilities and a desalination facility.We’re also pleased to get clarifying legislation to exempt water and wastewater utilities from the 2018 New Jersey tax law changes. We continue to have strong regulated customer growth. To date, we have welcomed about 16,500 new customer connections through closed acquisitions and organic growth. We have an additional 56,000 customer connections under agreement for acquisition. Walter will give you more detail shortly.Our adjusted earnings increase of over 11% was also driven by growth in our market-based businesses. We’re pleased to share that our homeowners services financial results to date are above projections with both excellent operating performance and Pivotal integration expenses favorable compared to our plan. We look forward to fully integrating the business and realizing synergies in the future.Our legacy Homeowner Services Group has had several wins this year. Most recently we announced formal partnerships with the city of Philadelphia and Fort Wayne, Indiana. We also have received notices from San Francisco and Toledo, Ohio of their preliminary intent to award partnerships to us following final negotiations.Additionally, we were pleased to announce last month that our Military Services Group won a 50 year, $591 million contract to serve Fort Leonard Wood army base in Missouri. We are very proud to grow to 14 basis, where we serve quality water, fire protection and sanitation services to the incredible men and women and their families who serve our nation and defend our liberties. And as we have already shared, we sell the majority of our Contract Services Group to Veolia in July. We are recognizing a positive $0.06 of benefit from that sale this quarter as a non-GAAP item.',
                 'AMP': "I will now turn the call over to Alicia Charity. You may begin.Alicia A. Charity - Ameriprise Financial, Inc.Thank you, Operator, and good morning. Welcome to Ameriprise Financial's third quarter earnings call. On the call with me today are Jim Cracchiolo, Chairman and CEO; and Walter Berman, our Chief Financial Officer. Following their remarks, we'll be happy to take your questions.Turning to our earnings presentation materials that are available on our website, on slide 2, you will see a discussion of forward-looking statements. Specifically during the call, you will hear reference to various non-GAAP financial measures which we believe provide insights into the company's operations. Reconciliations of non-GAAP numbers to their respective GAAP numbers can be found in today's materials.Some statements that we make on this call may be forward-looking, reflecting management's expectations about future events and overall operating plans and performance. These forward-looking statements speak only as of today's date and involve a number of risks and uncertainties. A sample list of factors and risks that could actual results to be materially different from forward-looking statements can be found in our third quarter 2018 earnings release, our 2017 Annual Report to shareholders, and our 2017 10-K report. We make no obligation to update publicly or revise these forward-looking statements.Turning to slide 3, you see our GAAP financial results at the top of the page for the third quarter. Below that, you see our operating results, followed by operating results excluding unlocking, which management believes enhances the understanding of our business by reflecting the underlying performance of our core operations and facilitates a more meaningful trend analysis. In the third quarter, we completed our annual unlocking. The comments that management makes on the call today will focus on operating financial results excluding unlocking.And with that, I'll turn it over to Jim.James M. Cracchiolo - Ameriprise Financial, Inc.Hello and thank you for joining our earnings call. This morning, we'll discuss our strong results, give you an update on the business, and our areas of focus today and going forward. Reflecting on the quarter, the macroeconomic and market picture has been positive for our clients and Ameriprise. The U.S. economy is strong, and that was punctuated by the Fed's decision to raise short-term interest rates again.However, Europe is showing signs of slower growth. Global trade issues and political uncertainty remain, including in the UK, especially as the deadline for Brexit nears. Recently, global equity markets have been more volatile. We're navigating that volatility and are very much focused on our clients.At the same time, consumer sentiment is strong. U.S. household wealth has reached a new high of over $100 trillion, which represents a significant long-term growth opportunity for Ameriprise.Turning to the company, over the years, we've built a strong business. We're well positioned based on our broad advice capability and our integrated model allows us to leverage resources and generates consistent results through market cycles. Ameriprise delivered another strong quarter. On an adjusted operating basis, excluding unlocking, revenue increased nicely, up 5%.We continue to generate excellent EPS growth, the third consecutive quarter of 20% growth or better, building on last year's strong results. And our return on equity was also very strong, now at 32%.I am pleased with the quarter and our long-standing track record for delivering results of this caliber. In addition, our assets under management and administration are now over $900 billion. As you know, our business generates significant free cash flow and we're disciplined in our capital allocation. We continue to shift our earnings mix to less capital-intensive business lines, with more than 70% of pre-tax adjusted operating earnings coming from our Wealth Management and Asset Management businesses in the quarter.At various investment forums over the years, some of you have asked us when we would grow our less capital-intensive business lines to get to 50% or even 60%. Today, we're over 70%. And we have the ability to continue to invest for growth and return to shareholders at a meaningful level.Let's move to some business highlights. Very clearly, we know the mass affluent and affluent want to work in a personal advice-based relationship with a trusted advisor. And these investors need either more advice as their assets grow, their lives become more complex, and markets experience volatility. At Ameriprise, we're positioned very well to serve this growing need. We're focused on serving more investors in our target market, those with $500,000 to $5 million, and we're seeing good organic growth at the $1 million-plus level.With good client growth and positive markets, Ameriprise client assets increased 9% to $588 billion. Our investment advisory platform is one of the largest in the industry. Net inflows into fee-based investment advisory accounts were $5.7 billion in the quarter, the fifth consecutive quarter of flows over $5 billion. And year-to-date, wrap flows are 14% higher than 2017, which was an exceptionally strong year.",
                 'ABC': "I would now like to turn the call over to your first speaker, Mr. Bennett Murphy. Please go ahead.Bennett Murphy - AmerisourceBergen Corp.Thank you. Good morning and thank you all for joining us for this conference call to discuss the AmerisourceBergen fiscal 2018 third quarter financial results. I am Bennett Murphy, Vice President, Investor Relations for AmerisourceBergen, and joining me today are Steve Collis, Chairman President and CEO; and Tim Guttman, Executive Vice President and CFO.On today's call, we will be discussing non-GAAP financial measures, which we use to assess the underlying performance of our business. The GAAP to non-GAAP reconciliations are provided in today's press release as well as on our website. During the conference call, we will also make forward-looking statements about our business and financial expectations on an adjusted non-GAAP basis, included but not limited to EPS, operating income and income taxes.Forward-looking statements are based on management's current expectations and are subject to uncertainty and change. AmerisourceBergen assumes no obligation to update any forward-looking statements or information and this call cannot be rebroadcast without the express permission of the company. We remind you that there are uncertainties and risks that could cause our future actual results to differ materially from our current expectations.For a discussion of key risk factors and other cautionary statements and assumptions, we refer you to our SEC filings including our most recent Form 10-K and to today's press release. I would like to remind you that we have posted a slide presentation to accompany this morning's press release. You can find it at our website, investor.amerisourcebergen.com.You will have an opportunity to ask questions after today's remarks by management. We do ask that you limit your questions to one per participant in order for us to get to as many participants and inquiries as we can within the hour.With that, I'll turn the call over to Steve. Steve?Steven H. Collis - AmerisourceBergen Corp.Thank you, Bennett, and good morning to everyone on today's call. Our continued focus on operational excellence helped us deliver a strong third quarter with revenue increasing 11% to $43 billion and adjusted diluted EPS growing 8% to $1.54. Our ability to consistently execute and maximize our opportunities for growth has helped us successfully navigate an evolving market landscape and overcome internal challenges, including the lower than expected contribution from PharMEDium this year, putting us in a strong position to finish fiscal 2018 on track and within our guidance range.We believe our results demonstrate the value that our businesses are creating for our manufacturer and provider partners and reinforce our long-term prospects for growth despite any near-term uncertainty. The importance of our work and our ability to consistently execute was on full display across our businesses during the third quarter. Over the past several weeks and months, many people have asked about the landscape for our industry and questioned our ability to adapt if there are changes to the model. In times like these I believe it's important to take a step back and look at our results in the context of the value we provide and the critical work we've already done to innovate in a fast changing market.In our Pharmaceutical Distribution and Strategic Global Sourcing group, we saw continued performance and execution in the core distribution business, both in increasing volumes with our existing customers and supporting their growth. I'm excited and pleased to announce that our team has re-signed of our key anchor customers, Humana. We've had a long relationship with this strategic partner and this early renewal shows a mutual appreciation and strong desire to continue growing together.Consistent with our customer contract rebalancing efforts, we were able to build a long term contract that is a win-win for both partners and one that does not create a margin headwind. AmerisourceBergen continues to benefit from our history of strategic investment and leadership in the distribution of specialty products.We have the best franchise, key customer relationships and continued support to launch our virtually all new specialty products. This quarter marks the 18th consecutive quarter with 10% of greater revenue growth for this part of the business. The connection we provide between manufacturers and our physician customers enables us to best support the commercialization of specialty products, both new innovative products and new biosimilar products, and empower those physicians to best serve their patients.As a brief update on PharMEDium, I'm proud of the execution of our associates in remediating at PharMEDium's Memphis facility. You'll recall we voluntarily suspended operations in December. We've been in active communication with the FDA and two weeks ago we notified the FDA of our intent to resume limited production at the Memphis facility and commence commercial distribution this month. We expect production in Memphis to increase gradually over time and to be fully operational in fiscal 2019.",
                 'AME': "It is now my pleasure to introduce Vice President of Investor Relations, Mr. Kevin Coleman. Please go ahead, sir.Kevin C. Coleman - AMETEK, Inc.Great. Good morning. Thank you, Andrew. Good morning, and thank you all for joining us for AMETEK's third quarter earnings conference call. With me this morning are Dave Zapico, Chairman and Chief Executive Officer; and Bill Burke, Executive Vice President and Chief Financial Officer.AMETEK's third quarter results were released earlier this morning and are available electronically on market systems and on our website in the Investors section of ametek.com. This call is also being webcasted and can be accessed on our website. The webcast will be archived and made available on our site later today.Any statements made by AMETEK during the call that are not historical in nature are to be considered forward-looking statements. As such, these statements are subject to change based on various risk factors and uncertainties that may cause actual results to differ significantly from expectations. A detailed discussion of the risk and uncertainties that may affect our future results is contained in AMETEK's filings with the SEC. AMETEK disclaims any intention or obligation to update or revise any forward-looking statements.Please note that during today's call, references will be made to some financial results on an adjusted basis. Please refer to the Investors section of ametek.com for a reconciliation of any non-GAAP financial measures used during this call. We'll begin today's call with prepared remarks by Dave and Bill, and then open up for questions.I'll now turn it over to Dave.David A. Zapico - AMETEK, Inc.Thank you, Kevin, and good morning, everyone. AMETEK had another spectacular quarter, as our businesses delivered exceptional results across all key operating and financial metrics. In the quarter, double-digit sales growth was driven by another excellent quarter of organic growth. We delivered significant operating margin expansion and robust earnings growth, while again increasing our full-year earnings guidance.We generated a record level of free cash flow and announced that we deployed $565 million on two highly strategic acquisitions. And lastly, given the strength of our acquisition pipeline, we announced an increase on our revolving credit facility to $1.5 billion, providing us added flexibility as we execute on our acquisition strategy. So overall, another outstanding quarter, reflecting the strength of the AMETEK business model, the differentiated nature of our businesses, and the excellent work of the entire AMETEK team.Now, on to the financial and business highlights. Total sales in the third quarter were $1.19 billion, up 10% compared to the third quarter of 2017. Organic sales growth was again very strong at 7%, with acquisitions adding 3% and foreign currency neutral to sales in the quarter.Growth remains broad-based across our businesses, with both our reportable segments growing 7% organically. This organic growth also remains very strong and balanced geographically, with Asia growing 12%, Europe 7%, and the U.S. 6% in the quarter.EBITDA in the third quarter was $312 million, up 14% over the third quarter of 2017, with EBITDA margins a very strong 26.1%. Third quarter operating income was $265.3 million, up 15% over the prior year. Reported operating income margins were up 100 basis points to 22.2%. Excluding the dilutive impact from acquisitions, operating margins increased by 130 basis points over the third quarter of last year. Third quarter earnings were $0.82 per diluted share, an increase of 24% over the same period last year, exceeding our guidance of $0.76 to $0.78 per share.Now, turning to the individual operating groups. First, the Electronic Instruments Group. EIG had a great quarter, with strong growth and exceptional operating performance. Overall sales for EIG in the third quarter were $742 million, up 10% over the same quarter of 2017. Organic sales were up 7%, with recent acquisitions contributing 4%. Foreign currency was a slight headwind to sales in the quarter.Organic growth remains broad-based across our EIG businesses. We saw very strong growth across our process businesses, including Rauland, which has experienced tremendous growth since being acquired in early 2017. Additionally, our Ultra Precision Technologies division had another outstanding quarter, with mid-teens organic sales growth.Operating income for EIG in the quarter was $190.3 million, up 17% over the prior year period. Reported operating income margins were excellent at 25.6%, up a very strong 130 basis points over last year's third quarter. Excluding the dilutive impact of acquisitions, EIG margins were up an outstanding 190 basis points over the same quarter of 2017.The Electromechanical Group also had an excellent quarter, with impressive sales growth and margin expansion. EMG sales in the third quarter were $450.9 million, a 9% increase over the same quarter last year. Organic sales growth was very strong, up 7% versus the prior year. The acquisition of FMH Aerospace contributed an additional 2% and foreign currency had no impact on sales. Sales growth remains strong and balanced across each of our key EMG businesses, including automation, aerospace and defense, and engineered materials.",
                 'AMGN': "I would now like to introduce Arvind Sood, Vice President of Investor Relations. Mr. Sood, you may now begin.Arvind SoodExcellent. Thanks Ian. Good afternoon, everybody. Thanks for joining us today. We have a lot of ground to cover, so I'll keep my comments brief.Continued execution, launch progress, and pipeline advancement are some key themes that come to mind as I think about our third quarter results. To elaborate on these themes and more, I'm joined today by Bob Bradway, our Chairman and CEO. After Bob's strategic review, our CFO David Meline will review our financial results for the third quarter and provide updated guidance for 2018. Also joining us today is Tony Hooper, and then you'll get to hear from Murdo Gordon, our newly appointed Head of Global Commercial Operations. Following Murdo's review of product performance, our Head of R&D, David Reese will provide a pipeline update. We will use slides to guide our discussion today and a link to those slides was sent separately.My customary reminder that we will use non-GAAP financial measures in today's presentation and some of the statements will be forward-looking statements. Our 10-K and subsequent filings identify factors that could cause our actual results to differ materially.So, with that, I would like to turn the call over to Bob.Robert BradwayOkay Arvind, thank you.On today's call I'll provide an overview of our performance in Q3, speak to the progress we're making in delivering our strategy for long-term growth, and share some thoughts on the current healthcare environment.Let me start by acknowledging that while we're in the midst of a period of high volatility driven by a variety of macro and political factors, our fundamental objectives of innovating for the benefit of patients and delivering for shareholders remain intact and unchanged.Looking to the future, there are bound to be headwinds but we're confident in our ability to navigate them from a position of strength. You see evidence of that strength reflected once again this quarter in our healthy balance sheet, strong cash flows, and efficient cost structure.We've said for some time that we expected growing pressures on drug prices in our industry. We can all see that plainly today. With price under pressure, having innovative products which can deliver volume growth by meeting the needs of large numbers of patients is ever more important.We have several such products and you can see the benefit of this in the third quarter where our new and recently launched products had double-digit unit volume growth. We believe differentiated products like Prolia, Repatha and Aimovig offer attractive long-term growth prospects.Prolia continues to build strength globally and there continues to be large untapped potential in the area of osteoporosis. With Repatha, we're taking steps to open up access and improve patient affordability to this important therapy.Our announcements this past week are significant and reflect our commitment to lower out-of-pocket costs for patients especially in Medicare in order to help ensure that patients who need Repatha get Repatha.Our new product story continues to unfold with Aimovig and migraine prevention. I'm sure it's not lost in you that Aimovig is off to a very strong start and in fact is shaping up to be one of the industry's most successful recent launches reflecting the pent-up demand that exists in this area.People suffering from migraine and their physicians have been waiting for years for an effective new therapy and they've reacted very well to Aimovig. We believe biosimilars can be an important growth driver for us as well as we’re now launching KANJINTI, a biosimilar Herceptin and AMGEVITA, a biosimilar to Humira internationally.Our innovative pipeline is moving very quickly. As you know, we aim to allocate capital to R&D where we believe we have innovative first-in-class molecules with the potential for large effect sizes and serious diseases. Over the past few months, we've introduced seven such programs into the clinic. Dave Reese will talk more about these in a moment.From a capital allocation standpoint, we've continued with our share buyback and dividend increases and will continue to take a disciplined approach to business development where we believe we can create value for Amgen's shareholders.Turning to the political policy environment for drugs in the U.S. this obviously remains very topical we don't expect this to end anytime soon. We remain committed to working with the administration and elected officials to advanced market-based reforms that will promote competition and improve access to new therapies without undermining our nation's innovative ecosystem.The U.S. leads the world and discover new cures and treatments for serious diseases. We believe our society benefits from embracing this innovation. If our objective is to lower healthcare costs and improve population health and productivity, what we need is more innovation and not less and we need a system that ensures access to it.Let me end by thanking our staff for their efforts this past quarter. I know they're working hard to finish the year on a strong note and I'm grateful to their dedication and for their dedication or for their dedication to our mission.",
                 'APH': "I would now like to introduce today's conference host, Mr. Craig Lampo. Sir, you may begin.Craig A. Lampo - Amphenol Corp.Thank you. Good afternoon, everyone. This is Craig Lampo, Amphenol's CFO and I'm here together with Adam Norwitt, our CEO. We would like to welcome you to our third quarter 2018 conference call. Our third quarter results were released this morning. I will provide some financial commentary on the quarter, and then Adam will give an overview of the business as well as current trends, and of course we will take questions.As a reminder, we may refer in this call to certain non-GAAP financial measures and may make certain forward-looking statements, so please refer to the relevant disclosures in our press release for further information.The company closed the third quarter with record sales of $2.129 billion and with record GAAP and adjusted diluted EPS of $1.01 and $0.99 respectively, exceeding the high end of the company's guidance for sales by approximately $110 million and adjusted diluted EPS by $0.06.Sales were up 16% in U.S. dollars and up 17% in local currency as compared to the third quarter of 2017. From an organic standpoint excluding both acquisitions and currency, sales in the third quarter increased 15%. Sequentially, sales were up 7% in U.S. dollars and 9% in local currency and organically.Breaking down sales into our two segments, our Cable business which comprised 5% of our sales was flat in U.S. dollars and up 4% in local currency as compared to the third quarter of last year. And our Interconnect business which comprised 95% of our sales was up 17% in U.S. dollars from last year, driven primarily by organic growth. Adam will comment further on trends by market in a few minutes.Operating income was $444 million for the third quarter. And operating margin was a record 20.9% in the third quarter of 2018, up 40 basis points compared to the third quarter of 2017 of 20.5%, and up 30 basis points compared to the second quarter of 2018 of 20.6%.From a segment standpoint in the Cable segment, margins were 13.1%, which is flat compared to the third quarter of 2017. And in Interconnect segment, margins were a strong 22.7% in the third quarter of 2018, which is compared to the third quarter of last year at 22.4%.This excellent performance is direct result of the strength and commitment of the company's entrepreneurial management team, which continues to foster a high-performance action-oriented culture in which each individual operating unit is able to appropriately adjust to market conditions and thereby maximize both growth and profitability in a dynamic market environment.Through the careful fostering of such a culture and the deployment of these strategies to both existing and acquired companies, our management team has achieved industry-leading operating margins and remains fully committed to driving enhanced performance in the future.Interest expense for the quarter was $25 million, which is comparable to last year. The company's adjusted effective tax rate was approximately 25.5% for the third quarter of 2018 compared to 26.5% in the third quarter of 2017. The adjusted effective tax rate in both periods excludes the excess tax benefit associated with stock option exercises as we have previously discussed.The company's GAAP effective tax rate for the third quarter of 2018 including the excess tax benefit associated with stock option exercises was approximately 23.8% compared to 21.8% in the third quarter of 2017. Adjusted net income was a strong 15% of sales in the third quarter of 2018.On a GAAP basis, diluted EPS grew 15% in the third quarter of this year to $1.01 from $0.88 in the third quarter of last year. Adjusted diluted EPS which excludes the excess tax benefit from stock option exercises in both periods grew 19% to a record $0.99 in the third quarter of this year from $0.83 in the third quarter of 2017 which also excluded certain acquisition-related expenses.Orders for the quarter were a record $2.12 billion, a 14% increase over the third quarter of last year resulting in a book-to-bill ratio of 1:1. The company continues to be an excellent generator of cash. Cash flow from operations was $339 million in the quarter or approximately 110% of adjusted net income.From a working capital standpoint; inventory, accounts receivable and accounts payable were approximately $1.2 billion, $1.7 billion and $1 billion respectively at the end of September. And inventory days, days sales outstanding and payable days were 78 days, 73 days and 63 days respectively which were all within our normal range.The cash flow from operations of $339 million along with stock option proceeds of $72 million, short-term borrowings of $19 million, and cash, cash equivalents and short-term investments on hand of approximately $29 million net of translation were used primarily to repay a net amount of $194 million under our commercial paper programs, to fund net capital expenditures of $72 million, to fund dividend payments of $69 million, to purchase approximately $35 million of the company's stock, and to fund acquisitions of and dividends paid to non-controlling interests of $11 million.During the quarter, the company repurchased 400,000 shares of stock at an average price of approximately $87 under the $2 billion three-year open-market stock repurchase plan. At September 30, cash and short-term investments were approximately $1 billion, the majority of which is held outside of the U.S.",
                 'APC': "I would now like to turn the conference over to Robin Fielder. Please go ahead, ma'am.Robin H. Fielder - Anadarko Petroleum Corp.Good morning, everyone. We're glad you could join us today for Anadarko's third quarter 2018 conference call.I'd like to remind you that today's presentation includes forward-looking statements and certain non-GAAP financial measures. We believe our expectations are based on reasonable assumptions. However, a number of factors could cause results to differ materially from what we discuss. We encourage you to read our full disclosure on forward-looking statements in our SEC filings and the GAAP reconciliations located on our website and attached to yesterday's earnings release. Additionally, we have provided additional detail in our website in the second quarter operations report.And with that, I'll turn it over to Al for some opening remarks.Robert A. Walker - Anadarko Petroleum Corp.Thanks, Robin, and good morning. Happy Halloween – and no, we are not in costumes, but we did consider it.Last year at this time, we discussed with you our intent to be increasingly capital efficient in a market that was and would likely continue to be volatile for oil price discovery. We expressed our focus on allocating capital to deliver healthy growth on a per debt-adjusted share basis, with the expectation of generating significant free cash flow as the commodity outlook improved.We also believed our material, scalable asset footprint could deliver attractive returns from our oil-weighted opportunities. Taken together, this growth per debt-adjusted share producing attractive cash flow return characteristics would form the basis for our capital allocation and capital return strategy for years to come.During the third quarter, we generated almost $550 million of adjusted free cash flow, as our margins increased to their highest level in more than four years at almost $34 per barrel. Meanwhile, we delivered $625 million of cash returns to shareholders, which is 38% of our third quarter cash flow from operations, including $500 million of additional share repurchases and a dividend payout of $125 million.The improving margins continue to be supported by our commitment to increasing our liquids mix, aided by our ability to access Gulf Coast markets for our domestic crude oil production. Currently, approximately 55% of our total oil volumes benefit from waterborne pricing. And by the end of next year with the startup of the Plains Cactus II line, we expect that to increase towards 70%.Over the last 12 months, we have repurchased more than 10% of our shares outstanding, increased our dividend by 400%, and announced plans to reduce debt by $1.5 billion. At the current strip, we expect to generate strong free cash flow during the fourth quarter and into 2019, given our continued commitment to investing our capital in a $50 oil price environment that provides healthy oil growth within this lower price cash flow assumption. We plan to use our cash and expected free cash flow to complete the remaining $500 million of our share repurchase program by the first half of 2019 while retiring an additional $1.4 billion of debt.We updated our full-year 2018 production guidance to tighten our previous range to account for the limited days left in the year as well as acknowledging that we did not need to seek significantly more oil growth over the balance of this year to achieve our annual objective. These changes also reflect the expected impact of hurricane-related downtime and previous capital allocation adjustment.We expect at this time to deliver in 2018 more than 13% oil growth, or 15% or more on a per debt-adjusted share basis, and a greater than 19% return for our cash flow on invested capital. These performance metrics were central to our investment strategy for the year.As we look ahead to 2019, we will continue to use the $50 oil price environment assumption to produce healthy multiyear double-digit oil growth, remain committed to returning capital to shareholders above this breakeven threshold, and demonstrate the durability of this investment strategy due to the strength and capital efficiency of our capital asset portfolio.Later this quarter, we will be announcing our 2019 investment plan, the timing of which will follow the November 6 midterm election and in particular the voting result in Colorado.Regardless of what happens, we feel confident we can continue to deliver our expected 2019 result given the flexibility of our portfolio. Multi-well pad and campaign-style development is beginning in the Delaware Basin, thanks to the tremendous work our teams have done to build out the necessary infrastructure and secure takeaway to facilitate future growth.",
                 'ADI': "I'd like to now introduce your host for today's call, Mr. Michael Lucarelli, Director of Investor Relations. Sir, the floor is yours.Michael LucarelliThank you, Jennifer and good morning, everybody. Thanks for joining our third quarter 2018 conference call. With me on the call today are ADI's CEO, Vincent Roche; and ADI's CFO, Prashanth Mahendra-Rajah.For anyone who missed the release, you can find it and related financial schedules at investor.analog.com. This conference call is being webcast live and a recording will be archived in the Investor section of our website.Now on to the disclosures. The information we're about to discuss including our objectives and outlook includes forward-looking statements. Actual results may differ materially from these forward-looking statements as a result of various factors including those discussed in our earnings release and in our most recent 10-Q. These forward-looking statements reflect our opinion as of the date of this call. We undertake no obligation to update these forward-looking statements in light of new information or future events.Our commentary about ADI's third quarter financial results will also include non-GAAP financial measures which exclude special items. When comparing our third quarter results to our historical performance special items are also excluded from the prior quarter and year-over-year results.Reconciliations of these non-GAAP measures to the most typically comparable GAAP measures and additional information about our non-GAAP measures are included in today's earnings release and on our web schedules which we've posted under the Quarterly Results section at investor.analog.com.Okay. So with that, I'll turn it over to ADI's CEO, Vincent Roche. Vince?Vincent RocheThanks, Mike and good morning to everyone.Analog Devices well we had an outstanding fiscal third quarter. Our performance demonstrates the disciplined execution of our strategy as we continue to deliver value for our customers helping them meet their evolving needs in this dynamic market environment.It also reflects our ability to deliver strong results for our shareholders while investing in growth opportunities to expand market share, continuously innovate and strengthen our position for the long-term.In the quarter, revenues came in above the high end of our guided range. Our growth was once again driven by continued strength in our B2B markets, especially, in the industrial and communication sectors. These strong results were supported by yet another record quarter from our Linear Tech franchise.Operating margins expanded meaningfully compared to last year and non-GAAP EPS increased over 20% year-over-year and came in above the high end of our guidance. We also delivered very strong free cash flow generating $2.2 billion in free cash flow on a trailing 12 month basis translating to approximately 36% free cash flow margin.ADI's cash generating capabilities enabled us to achieve our 2x leverage ratio goal three quarters ahead of plan and we are pleased to announce that we have reinstated our share repurchase program. And given the long-term prospects of this business, our Board has authorized an additional $2 billion in share repurchases.Now before Prashanth goes deeper into our financial performance, I'd like to continue the discussion around key market trends that are shaping our industry and the steps that ADI is taking to seize these opportunities.On the last earnings call, I spoke about the transition to 5G for wireless communications and what it means for ADI. Today, I'd like to focus on Industry 4.0, which is driving the next evolution of innovation and investment in industrial automation.For many industries, the digital factory will bring greater supply chain efficiencies higher quality more flexibility and a safer workplace, all contributing to higher productivity. This manufacturing in which cyber physical systems monitor the physical process of the factory or plant will use additional data for decentralized decision making, human assistance, increased safety and more predictability. This will require a vast deployment of intelligent sensors at the edge.For many decades, ADI has been viewed as the go-to-edge solution provider, the place where the data is born so to speak. And I'll highlight later how we are enabling this evolution. Although ADI is excited about this emerging opportunity, we expect this transition to take time while the deployment of today's existing sensing signal processing and power architectures will continue to drive growth for many years.In fact, our core automation business represents a significant portion of our total industrial business today, and we're taking a long view to ensure that we keep at the cutting edge of technology and customer support.Our domain expertise is unmatched and gives us invaluable insights to understand our customers' challenges and where the markets are going. And we've been investing ahead broadening our portfolio through targeted R&D, collaborating with leading automation customers and leveraging strategic M&A to enhance our offerings to include algorithms and software.At the same time, we've extended our reach with the addition of the LTC sales force and SAEs. And the complementarity of our customer bases enables us to access more opportunities at current customers and to increase penetration at new customers. We believe that these targeted investments position us to continue to grow our market share and to outperform in the years to come.",
                 'ANSS': "At this time, I would like to turn the call over to Ms. Arribas for some opening remarks.Annette N. Arribas - ANSYS, Inc.Good morning, everyone. Our earnings release and the related prepared remarks documents have been posted on the homepage of our new and improved Investor Relations website this morning. They contain all of the key financial information and supporting data relative to our third quarter and year-to-date financial results and business update, as well as our updated Q4 and fiscal year 2018 outlook and the key underlying assumptions.I would like to remind everyone that in addition to any risks and uncertainties that we highlight during the course of this call, important factors that may affect our future results are discussed at length in our public filings with the SEC, all of which are also available via our website. Additionally, the company's reported results should not be considered an indication of future performance, as there are risks and uncertainties that could impact our business in the future. These statements are based upon our view of the business as of today and ANSYS undertakes no obligation to update any such information, unless we do so in a public forum.During this call, and in the prepared remarks, we'll be referring to non-GAAP financial measures unless otherwise stated. Please take any reference to revenue to mean revenue under ASC 606 unless we explicitly note that we're referring to ASC 605 results. Note that all references to growth will be in terms of ASC 605 results since we have no baseline for last year under ASC 606. A discussion of the various items that are excluded in a full reconciliation of GAAP to comparable non-GAAP financial measures under both ASC 605 and ASC 606 are included in this morning's earnings release materials and related Form 8-K.I would now like to turn the call over to our CEO, Ajei Gopal for his opening remarks. Ajei?Ajei S. Gopal - ANSYS, Inc.Thank you, Annette, and good morning, everyone. Q3 was yet another exceptional quarter. We exceeded the high end of our Q3 revenue and our earnings per share guidance by $8 million and $0.24 respectively. Our revenue growth as measured under ASC 605 was 12%, leading to record third quarter ASC 605 revenue and earnings per share. We recorded $762 million of deferred revenue and backlog as measured in the ASC 605, which represents a 14% year-over-year increase. Our ACV growth in constant currency was outstanding at over 13% for the quarter and 11% year-to-date. This reflects strong customer demand for ANSYS solutions and our ongoing success in the market.Given our robust performance to-date in 2018 and the strength of our pipeline going into the last quarter of the year, we are increasing our revenue, our EPS, and our operating cash flow guidance for the full year. We're also raising ACV guidance at the midpoint. Maria will provide more details in a few minutes.In Q3, we recorded numerous six figure deals across several major verticals. With the race to 5G in high gear, the high-tech vertical performed well with leading communications companies choosing ANSYS in part because of the multi-physics design capabilities enabled by our Chip-Package-System workflow. Our market-leading capabilities and enabling electrification and autonomy drove customer investments across the automotive sector. The aerospace and defense sector performed well as investment is increasing in the U.S. and Europe. The industrial equipment vertical, particularly the rotating machinery market, is also benefiting from the recovery in oil and gas.In Q3, we also saw an increase in investment from the healthcare industry. We work with healthcare leaders such as Medtronic for many years on the use of in silico medicine to advance medical device design. At the September meeting of the Avicenna Alliance, a global alliance of healthcare industries and researchers that was set up at the request of the European Commission, Medtronic reported that modeling and simulation helped them release the product to market two years earlier, treating 10,000 patients during this period, and saving an estimated $10 million.From a geographic perspective, ASC 605 revenues grew double-digits in the Americas and Europe, while Asia Pacific grew 9%, all in constant currency. Our three largest markets, the U.S., Japan and Germany, led our performance in the quarter each with double-digit revenue growth. Our new go-to-market strategy, which we described at Investor Day last year, is seeing success. Our approach enables us to effectively grow and close large enterprise deals, while efficiently addressing the large volume of our transactional business. We're intelligently matching our customer size, location and level of simulation sophistication with our routes to market, which include the use of strategic sales teams, territory sales, indirect channel partners and inside sales reps.",
                 'ANTM': "I would now like to turn the conference over to the company's management.Chris Rigg - Anthem, Inc.Good morning, and welcome to Anthem's third quarter 2018 earnings call. This is Chris Rigg, Vice President of Investor Relations. And with us this morning are Gail Boudreaux, President and CEO; John Gallina, our CFO; Pete Haytaian, President of our Commercial and Specialty Business Division; and Felicia Norwood, President of our Government Business Division.Gail will begin the call by giving an overview of our third quarter financial results, followed by commentary around our focus on execution and our enterprise-wide growth priorities. John will then discuss our key financial metrics in greater detail and go over our updated 2018 outlook. We will then be available for Q&A.During the call, we will reference certain non-GAAP measures. Reconciliations of these non-GAAP measures to the most directly comparable GAAP measures are available on our website at antheminc.com.We will also be making some forward-looking statements on this call. Listeners are cautioned that these statements are subject to certain risks and uncertainties, many of which are difficult to predict and generally beyond the control of Anthem. These risks and uncertainties can cause actual results to differ materially from our current expectations. We advise listeners to carefully review the risk factors discussed in today's press release and our quarterly filings with the SEC.I will now turn the call over to Gail.Gail Koziara Boudreaux - Anthem, Inc.Good morning, everyone. Thank you for joining us for Anthem's third quarter 2018 earnings call. Today, we reported third quarter 2018 GAAP earnings per share of $3.62 and adjusted earnings per share of $3.81, reflecting improved execution across our enterprise and growing momentum within our businesses.Anthem's third quarter operating revenue grew 4% over the prior-year quarter to $23 billion. Strong growth in our Government Business more than offset the planned reduction in our Individual business. Administrative fees and other revenue increased more than 15% over the prior-year quarter, reflecting increased sales of our Specialty products, clinical engagement programs, and growth of fee-based membership. Third quarter membership met expectations, and we expect growth to accelerate in the fourth quarter.Our third quarter medical cost performance was strong and reflects the impact of our value-based integrated care arrangement. Anthem's Whole Health Connection, which delivers clinical integration across our medical and Specialty product is closing gaps in care and reducing costs. Over the last year alone, this program has helped identify over 30,000 diabetic through vision exams.Our 2018 adjusted earnings per share outlook is now greater than $15.60, reflecting a 30% increase from 2017. Our revised 2018 guidance is a direct result of the enterprise-wide changes we made to improve execution, including sales force automation, the introduction of new clinical programs, and further scaling of existing assets like CareMore. We believe the investments and cultural changes we've implemented in 2018 provide a solid foundation for growth in 2019 and beyond.We're focused on creating innovative, meaningful, and cost-effective healthcare solutions for consumers. Our collaboration with Walmart, launching in January of 2019, gives our Medicare Advantage member the flexibility to use their over-the-counter plan allowances to purchase medications and health-related items from Walmart's 4,700 stores and online. Today, more than 90% of Americans live within 10 miles of a Walmart store, and our partnership demonstrates our commitment to driving convenient, affordable access to healthcare for our seniors.Our 2019 Medicare Advantage plan design feature competitive medical and pharmacy benefits and include significant enhancements to ancillary benefits designed to address whole person care. Overall, our 2019 Medicare Advantage offering position us well for both mid-double digit growth and stable margins.As we highlighted on the second quarter earnings call, momentum in our group Medicare Advantage business is accelerating, and we now expect total membership to exceed 150,000 in the first quarter of 2019. Our success in winning business is a result of our deep penetration in the commercial market, industry-leading Blue brand, and strong consumer relationships. As the trusted brand for many of the largest commercial accounts, we benefit from a captive new business pipeline and are uniquely-positioned for multiyear growth in the group Medicare segment.We're disappointed with the Star Quality rating scores that were recently announced by CMS, which will impact payments in 2020. As part of our overall Medicare Advantage Star Strategy, we're deploying incremental capital to improve clinical quality, pharmacy and medication management, and the member experience. We believe these steps will lead to an improved Stars outcome for the 2021 payment year.",
                 'AON': "Now it is my pleasure to turn the call over to Greg Case, CEO of Aon plc. Please go ahead, GregGregory CaseThanks, and good morning, everyone, and welcome to our third quarter conference call. Joining me today is our CFO, Christa Davies. In addition, we are excited to have our Co-Presidents, Eric Andersen and Mike O'Connor, join the call today. As highlighted last quarter, we intend to use our time on the call to provide you with more insight into the longer term view for the firm, with greater transparency into strategic initiatives. Also, like last quarter, we posted our detailed financial presentation on our website. We believe this approach provides a more thoughtful, long-term focus towards operational performance, including examples of investment in growth areas that are strengthening our capability to serve clients. Mike and Eric will be great additions to this discussion to give you another perspective on the power of leading Aon United.With that opening, we'd like to start by noting that our colleagues around the world delivered another strong result for the third quarter, with positive performance across each of our key metrics, including: 6% organic revenue growth; 190 basis points of margin expansion; 18% operating income growth; and 34% growth in EPS; and a similar strong performance across our key metrics year-to-date, reinforcing our continued long-term momentum, which Christa will discuss a little bit more in a bit.This continued momentum is a direct result of the strategic actions we've progressively taken to build upon the idea of leading Aon United and the impact it's having at scale on our business.As discussed last quarter, we've been laying the foundation for Aon United for over a decade, evolving our portfolio, investing in new content and capability and aligning client demand through programs like Aon Client Promise, all focused on increasing our relevance and strengthening our ability to deliver value for clients.More recently, over the last few quarters, we've taken steps to further reinforce and amplify this progress. In doing so, we've truly entered the era of Aon United, through structural changes that break down barriers and make it easier to deliver the best of the firm to clients: a single leadership team; a single P&L; a single brand; a single operating model. Most compelling, a united global professional services firm.Just last week, we announced another reinforcing step in this journey: The formation of our New Ventures Group, a team of leaders from across the firm who've created additional capacity to build on our already industry-leading record in innovation and further accelerate new sources of value for clients, led by Tony Goland as Chief Innovation Officer. The benefits that accrue when our global firm effectively works together are distinctive and unmatched in the industry and are unlocking significant value for our clients through greater innovation and superior, tailor-made solutions that fit their specific business objectives. Let me give you 2 quick examples. The first is an example of how our strategic investment in client-serving capabilities led to connections across solution lines and geographies in our core business that created unique client value. And the second is an example of how we're expanding the marketplace by attacking new business challenges on behalf of our clients.The first example. I was recently with one of our enterprise client leaders, which is a team of business advisers that coordinates Aon's value proposition for a select group of significant clients and a large private equity client. And we were discussing the needs of their portfolio companies. And in doing so, it became clearer that they wanted to put their balance sheet to work in new ways and that they aspired to create value outside of traditional PE opportunities. Our enterprise client leader connected that aspiration to our pension de-risking capabilities, and we began a conversation about the opportunity to apply their capital to help de-risk the pension obligations of their portfolio companies and potentially, beyond that, in the broader marketplace. By doing so, we expanded the conversation from risk mitigation to value creation. This is likely to be a new source of value for the client over the coming years and was made possible by the connection our enterprise client leader made between our commercial risk solutions capabilities and related offerings in Retirement Solutions. This example also demonstrates how the enterprise client group could help us create a templated approach to value creation and that we can replicate across sectors. And more broadly, how that, that team is helping us institutionalize our approach to Aon United growth opportunities and developing best practices designed to make it easier for all of our client-facing colleagues to bring the best of our firm to clients.This second example is all about marketplace expansion, as we continue our focus on expanding the marketplace by developing solutions to our clients' most pressing business needs. In Q3, we continued to invest behind cyber, with the launch of our silent cyber solution, driven by analytics and backed by an evolving reinsurance solution to help carriers respond to expanding cyber risk and regulations.",
                 'AOS': "I'd now like to turn the conference over to your host, Patricia Ackerman, Vice President of Investor Relations and Treasurer. Please go ahead.Patricia K. Ackerman - A. O. Smith Corp.Thank you, Adam. Good morning, ladies and gentlemen, and thank you for joining us on our 2018 third quarter results conference call. With me participating in the call are Ajita Rajendra, Executive Chairman; Kevin Wheeler, Chief Executive Officer; and John Kita, Chief Financial Officer.Before we begin with Kevin's remarks, I would like to remind you that some of the comments that will be made during this conference call, including answers to your questions, will constitute forward-looking statements. These forward-looking statements are subject to risks that could cause actual results to be materially different. Those risks include, among others, matters that we have described in this morning's press release.Also, as a courtesy to others in the question queue, please limit yourself to one question and one follow-up per turn. If you have multiple questions, please rejoin the queue.I will now turn the call over to Kevin, who will begin our prepared remarks on slide 3.Kevin J. Wheeler - A. O. Smith Corp.Thank you, Pat, and good morning, ladies and gentlemen. Here are a few comments about our third quarter. We achieved sales of $754 million, net earnings of $0.61 per share or 13% higher than our earnings per share in 2017. We continued to review our capital allocation and dedicated portion of our cash to return to shareholders. We repurchased 1.7 million shares for approximately $106 million through the first nine months of the year. We plan to continue buying back our shares at a previously stated $135 million annual pace using a 10b5-1 plan. We expect to opportunistically buy back up to $65 million worth of shares in the open market in 2018.We announced a 22% increase to our dividend in October. This is the second increase this year. The five-year compounded annual growth rate of our dividend is about 30%. We repatriated nearly $300 million during the first nine months of 2018 using the proceeds to repurchase our shares and pay down floating rate debt. Our A.O. Smith-branded water treatment products are now in 1,700 Lowe's stores nationwide.John will now describe our results in more detail beginning on slide 4.John J. Kita - A. O. Smith Corp.Sales for the third quarter of $754 million were 1% higher than the same quarter in 2017. Net earnings in the third quarter of $105 million increased 12% from the third quarter in 2017. Third quarter earnings per share of $0.61 increased 13% compared with the same quarter in 2017. Sales in our North American segment of $487 million were flat compared with the third quarter of 2017. Pricing actions in 2018 on water heaters and boilers related to higher steel and freight costs were more than offset by lower water heater volumes.North America water treatment sales comprised of Aquasana, Hague and our recently-launched water treatment products at Lowe's, incrementally added $9 million to our North America segment sales. Rest of World segment sales of $274 million increased 1% compared with the same quarter in 2017. Sales – China sales growth was 2.5% in local currency. Higher sales of water treatment products and air purifiers in China were partially offset by a decline in electric water heaters compared with the prior year. Currency translation reduced sales by approximately $6 million compared with the third quarter 2017.On slide 6, North America segment earnings of $106 million were 4% lower than segment earnings in the same quarter in 2017. The unfavorable impact from lower sales of water heaters and higher steel and freight costs were partially offset by pricing actions. Spending associated with the launch of water treatment products at Lowe's amounted to approximately $2 million. As a result of these factors, North-American segment margin of 21.7% was lower than last year.Rest of World earnings of $39 million increased nearly 16% compared with third quarter of 2017, primarily as a result of lower expenses associated with employee incentive programs and smaller losses in India. As a result, third quarter segment margin of 14.3% was significantly higher than one year ago.Our corporate expenses were higher than last year as a result of several miscellaneous items. Our effective income tax rate in the third quarter of 2018 was 20.5%. The rate was lower than the 28.8% experienced during the third quarter last year, primarily due to lower federal income taxes related to tax reform. The lower effective income tax rate benefited third quarter 2018 earnings by $0.06 per share.",
                 'APA': "Gary Clark, Investor Relations, you may begin your conference.Gary T. Clark - Apache Corp.Good morning and thank you for joining us for Apache Corporation's Third Quarter 2018 Financial and Operational Results Conference Call. We will begin the call today with an overview by Apache's CEO and President, John Christmann. Tim Sullivan, Executive Vice President of Operations Support, will then provide additional operational color. Steve Riney, Executive Vice President and CFO, will summarize our third quarter financial performance. Also available on the call to answer questions are Apaches Senior Vice Presidents: Brian Freed, Midstream and Marketing; Mark Meyer, Energy Technology, Data Analytics and Commercial Intelligence; and Dave Pursell, Planning Reserves and Fundamentals.Our prepared remarks will be approximately 25 minutes in length, with the remainder of the hour allotted for Q&A. In conjunctions with yesterday's press release, I hope you have had the opportunity to review our third quarter Financial and Operational Supplement, which can be found on our Investor Relations website, at investor.apachecorp.com.On today's conference call we may discuss certain non-GAAP financial measures. A reconciliation of the differences between these non-GAAP financial measures and the most directly comparable GAAP financial measures can be found in the supplemental information provided on our website. Consistent with previous reporting practices, adjusted production numbers cited in today's call are adjusted to exclude non-controlling interest in Egypt and Egypt's tax barrels.Finally, I would like to remind everyone that today's discussions will contain forward-looking estimates and assumptions based on our current views and reasonable expectations. However, a number of factors could cause actual results to differ materially from what we discussed today. A full disclaimer is located with the supplemental data on our website.And with that, I will turn the call over to John.John J. Christmann IV - Apache Corp.Good morning and thank you for joining us. On the call today, I will discuss Apache's strategic positioning, provide a preview of 2019, comment on our strong third quarter performance and review our key focus areas. Apache is a very different and much improved company from four years ago when oil prices began their prolonged downturn in the fall of 2014. We acknowledged very early on that the industry including Apache needed to make significant changes, not only in terms of reducing activity levels and overall cost structure, but equally to reestablish long-term returns discipline in the capital program.At that time, we chose to significantly curtail our drilling program, allowing production to decline rather than pursue growth in an environment where commodity prices and costs were not properly aligned. We refrained from participating in high-cost acreage acquisitions in the heart of proven plays, choosing instead to build our unconventional exploration capabilitiesAn important outcome of this strategy was the discovery of Alpine High. We have frequently stated our philosophy that an E&P company should be capable of living within cash flow from operations, generating sustainable long-term reserve and production growth while also returning capital to shareholders.After significant upfront investment at Alpine High and the pending completion of our Altus Midstream transaction, Apache has turned the corner and is well positioned to deliver on this philosophy for many years to come. We continue to generate steady production growth on a flat activity set and are poised to deliver positive free cash flow in 2019. This can be sustained over the long term through development of our extensive inventory. It will be supplemented by organic exploration, including discoveries in hand today, and a refreshed portfolio of new opportunities.Lower F&D costs, increasing returns and continual portfolio hydrating will accompany our growth and drive sustainable shareholder value growth through time. This is the investment proposition Apache offers and it is one that we strongly believe in, as evidenced by our decision to begin repurchasing shares in September under an existing authorization. Recently, our Board of Directors approved a new authorization for the repurchase of 40 million shares, which represents more than 10% of shares outstanding.Now I would like to provide a preview of 2019. For more than a year, Apache has operated at a relatively constant upstream activity level, which has enabled us to deliver operational efficiencies, effectively control costs and generate sustainable liquids production growth. We anticipate maintaining a similar activity level next year, but on a lower capital budget. If changes in expected cash flow dictate, we have the flexibility to reduce our activity levels accordingly.In 2019, assuming commodity prices in line with the current strip, Apache expects upstream capital investment of approximately $3 billion, which is consistent with our current guidance and is lower than 2018. Adjusted production at the high end of our 410,000 to 440,000 BOEs per day guidance range, representing more than 15% growth in the U.S. and 10% growth overall, positive free cash flow and continued return of capital to shareholders. This is well-aligned with the philosophy I outlined at the beginning of the call and we look forward to providing a more detailed 2019 outlook in February.",
                 'AIV': "I would now like to turn the conference over to Lisa Cohn, Executive Vice President and General Counsel. Please go ahead, ma'am.Lisa CohnThank you. Good day. During this conference call, the forward-looking statements we make are based on management's judgment, including projections related to 2018 results and 2019 expectations. These statements are subject to certain risks and uncertainties, a description of which can be found in our SEC filings. Actual results may differ materially from what may be discussed today.We will also discuss certain non-GAAP financial measures, such as AFFO and FFO. These are defined and are reconciled to the most comparable GAAP measures in the supplemental information that is part of the full earnings release published on Aimco's Web site.Prepared remarks today come from Terry Considine, our Chairman and CEO; Keith Kimmel, Executive Vice President, in charge of Property Operations; Wes Powell, Executive Vice President, in charge of Redevelopment; and Paul Beldin, our Chief Financial Officer. A question-and-answer session will follow our prepared remarks.I'll now turn the call over to Terry Considine. Terry?Terry ConsidineThank you, Lisa, and good morning to all of you on this call. Thank you for your interest in Aimco. The apartment business is good and Aimco had another solid quarter. In fact, the strength of our summer season led Paul to increase its expectations for full year bottom line for the second time in two months.We are upbeat as we finish 2018 and optimistic that 2019 will bring further good news. As we look forward to next year, we continue to look for accretive opportunities to deploy capital, while maintaining our commitment to a safe and flexible balance sheet. For their hard work and solid results in the third quarter and for the opportunities they provide us as we look forward, I offered great things to my Aimco teammates, both here in Denver and across the country.And now for more a detailed report in the third quarter, I'd like to turn the call over to Keith Kimmel, Head of Property Operations. Keith?Keith KimmelThanks, Terry. I'm pleased to report that we had a solid third quarter in operations. We've executed a purposeful strategy focused on average daily occupancy. Third quarter finished at 96.3%, 30 basis points better than the third quarter of 2017. Our occupancy accelerated each month throughout the quarter, from 96% in July to 96.6% in September. This has further grown to 96.8% in October, which we expect to maintain throughout the fourth quarter and provide a springboard into 2019.This is driven by our low turnover of 44.8%, resulting from consistently high customer satisfaction and focused customer selection. We continue to have pure leading margins in our same-store portfolio of 73.8% for the quarter, benefiting from our consistent focus on staffing efficiencies, process centralization and automation.Turning to our same store results, revenues were up 3.1% for the quarter. Consistent with our plan, expenses were up 4.5% due to a challenging comp related to non-recurring items in 2017 and higher investment in maintenance across our portfolio. Going forward, we see continued innovation enabling us to return to a more Aimco like expense result in 2019.Finally, net operating income was up 2.6%. Locking at leases which transacted in the quarter; new lease were up 2.2%; renewals were up 4.2%; and same-store blended lease rates were up 3.2%. We saw new lease rates of 4% to 5% in Los Angeles, Boston, the Bay Area and San Diego with the most pressure on new lease rates in Miami, where we saw the impact of directly competitive new supply.Turning to the third quarter same-store revenue growth; our top performers representing more than half of our same-store portfolio, had revenue increases over 4% for the quarter; this includes Boston, San Diego, the Bay Area, Miami and Los Angeles; our lower same-store revenue growth came from Atlanta and Philadelphia, which had negative revenue growth for the quarter; these two markets represent less than 5% of our same-store revenue. Outside of same-store, our premier Philadelphia redevelopment and acquisition communities have delivered strong results; the Sterling was occupied above 95% for the quarter.Today, the first three towers at Park Towne are 95% occupied and the fourth tower is now over 80% leased. We’ve accomplished this with rents in line with our underwriting. Our four acquisition communities in Philadelphia have achieved revenue on track with underwriting. And in Washington DC, we continue to be encouraged by the performance at our Bent Tree acquisition where rents were increased $100 on day one and we’ve seen occupancy rise by more than 300 basis points, and this has resulted in both rents and occupancy ahead of our underwriting.",
                 'AAPL': "Nancy Paxton - Apple, Inc.Thank you. Good afternoon and thanks to everyone for joining us. Speaking first today is Apple CEO Tim Cook and he'll be followed by CFO Luca Maestri. After that we'll open the call to questions from analysts.Please note that some of the information you'll hear during our discussion today will consist of forward-looking statements including without limitation those regarding revenue, gross margin, operating expenses, other income and expense, taxes, capital allocation, share repurchases, dividends and future business outlook. Actual results or trends could differ materially from our forecast.For more information, please refer to the risk factors discussed in Apple's most recently-filed periodic reports on form 10-K and form 10-Q and the form 8-K filed with the SEC today, along with the associated press release. Apple assumes no obligation to update any forward-looking statements or information which speak as of their respective date.I'd now like to turn the call over to Tim for introductory remarks.Timothy Donald Cook - Apple, Inc.Thank you, Nancy, and thanks to everyone for joining us. Today we're proud to report our best June quarter revenue and earnings ever thanks to the strong performance of iPhone, services and wearables. We generated $53.3 billion in revenue, a new Q3 record. That's an increase of 17% over last year's result, making it our seventh consecutive quarter of accelerating growth, our fourth consecutive quarter of double-digit growth and our strongest rate of growth in the past 11 quarters.Our team generated record Q3 earnings per share of $2.34, an increase of 40% over last year. We're extremely proud of these results and I'd like to share some highlights with you.First, iPhone had a very strong quarter. Revenue was up 20% year-over-year and our active installed base grew by double-digits, driven by switchers, first time smartphone buyers and our existing customers whose loyalty we greatly appreciate. iPhone X was the most popular iPhone in the quarter once again, with a customer satisfaction score of 98% according to 451 Research. Based on the latest data from IDC, iPhone grew faster than the global smartphone market, gaining share in many markets including the U.S., Greater China, Canada, Germany, Australia, Russia, Mexico and the Middle East and Africa.Second, we had a stellar quarter in services which generated all-time record revenue of $9.5 billion fueled in part by double-digit growth in our overall active installed base. We feel great about the momentum of our services business and we're on target to reach our goal of doubling our fiscal 2016 services revenue by 2020. Our record services results were driven by strong performance in a number of areas and I'd like to briefly mention just some of these.Paid subscriptions from Apple and third parties have now surpassed $300 million, an increase of more than 60% in the past year alone. Revenue from subscriptions accounts for a significant and increasing percentage of our overall services business. What's more, the number of apps offering subscriptions also continue to grow. There are almost 30,000 available in the App Store today.The App Store turned 10 years old this month and we set a new June quarter revenue record. The App Store has exceeded our wildest expectations, igniting a cultural and economic phenomenon that has changed how people work, learn and play. Customers around the world are visiting the App Store more often and downloading more apps than ever before. And based on third party research estimates, the App Store generated nearly twice the revenue of Google Play so far in 2018.The app economy is thriving and thanks to the App Store, it's generating jobs for tens of millions of people around the world. Our developers have earned over $100 billion from the App Store since its launch and we couldn't be more proud of them and what they've accomplished. We're hearing lots of developer excitement around our upcoming OS releases, which I'll talk about more in a moment, and can't wait to see what they can come up with next.We've experienced rapid growth in our App Store search ad service and as we announced earlier this month, we are expanding our geographic coverage to Japan, South Korea, France, Germany, Italy and Spain. We're also seeing strong growth in many of the other services as well. Just a few examples, Apple Music grew by over 50% on a year-over-year basis. AppleCare revenue grew at its highest rate in 18 quarters, partly due to our expanded distribution initiative. Cloud services revenue was also up over 50% year-over-year. Our communications services are experiencing record usage. We've hit all-time highs for both the number of monthly active users of Messages and the number of FaceTime calls made with growth accelerating from the March to June quarters.",
                 'AMAT': "I would now like to turn the conference over to Michael Sullivan. Please go ahead, sir.Michael SullivanGood afternoon. And thank you for joining us. I’m Mike Sullivan, Head of Investor Relations at Applied Materials. We appreciate you joining us for our third quarter of fiscal 2018 earnings call, which is being recorded. Joining me are Gary Dickerson, our President and CEO and Dan Durn, our Chief Financial Officer.Before we begin, let me remind you that today’s call contains forward-looking statements, including Applied’s current view of its industries, performance, products, share positions and business outlook. These statements are subject to risks and uncertainties that could cause actual results to differ materially and are not guarantees of future performance. Information concerning these risks and uncertainties is contained in Applied’s most recent Form 10-Q and 8-K filings with the SEC. All forward-looking statements are based on management’s estimates, projections and assumptions as of August 16, 2018, and Applied assumes no obligation to update them.Today’s call also includes non-GAAP financial measures. Reconciliations to GAAP measures are contained in today’s earnings press release and in our reconciliation slides, which are available on the Investor Relations page of our Web site at appliedmaterials.com.And now, I’d like to turn the call over to Gary Dickerson.Gary DickersonThanks Mike. I’m pleased to report that our revenue for the quarter was up 19% compared to the same period last year and the second highest in the Company's history. Fiscal 2018 remains on track to be another record setting year for Applied Materials, and we expect each of our major businesses to deliver strong double-digit growth.In today's call, I’ll start by providing our perspective on the market environment and our business performance. Then I’ll lay out our views on the industry's future growth drivers and describe how we’re evolving our strategy to take full advantage of the tremendous opportunities ahead. In aggregate, we see ongoing strength in our markets. Customers are making rational investments in new capacity resulting in well balanced supply demand dynamics. At the same time, they are aggressively pursuing their development roadmaps with healthy spending on next generation technologies. Demand for wafer fab equipment is on track to be an all-time record in 2018, and our view of 2019 remains positive, our thesis that spending in 2018 plus 2019 combined will exceed $100 billion remains firmly intact. The details within our 2018 forecast are consistent with the view we shared during our last call with the exception of a recent downward revision to our foundry outlook.As foundry customers optimize existing capacity, they’ve trimmed their capital spending plans for the year. They are still pushing forward with leading-edge developments, prioritizing current investment towards long lead time equipment, which is a positive leading indicator for 2019. NAND bit demand is expected to grow at about 40% this year with bit supply growing slightly faster. As a result, we see spending levels flat to modestly down from last year's record levels.DRAM investments are strong, up approximately 50% year-over-year as customers invest in capacity and technology to meet growing demand for high-performance DRAM for data centers. Capital investments by the leading cloud service providers continues to strengthen, up about 85% year to date compared to 2017. And in line with our prior view we also expect, logic investments to be higher this year.Stepping back and looking at the broader context, 2018 shows how the industry has fundamentally changed over the past five years. More diverse demand drivers spanning consumer and enterprise markets combined with very disciplined investment has reduced cyclicality. We’re not seeing the large fluctuations in wafer fab equipment spending that we did in the past. Over the same time period, we've also driven significant changes within Applied that have resulted in a larger, less volatile and more resilient business. In semiconductor, we've gained 7 points of market share in memory since 2013, while maintaining our traditionally strong position in logic foundry. As a result, we are now very well balanced across market segments.We’ve built the strong portfolio of products that addressed major technology inflections. For example, by developing tools for next generation multi patterning, we have grown our patterning business in DRAM, logic and foundry from about $100 million in 2013 to more than $1 billion this year. We expect our patterning opportunity to grow by another billion dollars as EUV and new materials enabled patterning steps are adopted over the next five years.In display, we have scaled the business from about $600 million in 2012 to approximately $2.5 billion this year. In both TV and mobile, customers are investing in new technologies and that plays directly to Applied’s strengths. We expect display to remain a powerful growth driver for the Company over the long-term.",
                 'APTV': "I would now like to turn the call over to Elena Rosman, Aptiv’s, Vice President of Investor Relations. Elena, you may begin your conference.Elena RosmanThank you, Albert. Good morning, and thank you to everyone for joining Aptiv’s third quarter 2018 earnings conference call.To follow along with today’s presentation, our slides can be found at ir.aptiv.com. Consistent with prior calls, today’s review of our actual and forecasted financials exclude restructuring and other special items and will address the continuing operations of Aptiv.The reconciliation between GAAP and non-GAAP measures for both our Q3 financials, as well as our outlook for the fourth quarter and full year are included at the back of the presentation and in the earnings press release.Please see slide two for disclosure on forward-looking statements, which reflects Aptiv’s current view of future financial performance, which may materially be different from our actual performance for reasons that we cite in our Form 10-K and other SEC filings.Joining us today will be Kevin Clark, Aptiv’s President and CEO; and Joe Massaro, CFO and Senior Vice President. Kevin will provide a strategic update on the business and then Joe will cover the financial results on our outlook for 2018 in more detail.With that, I would like to turn the call over to Kevin Clark.Kevin ClarkI'm going to begin by providing an overview of the third quarter. I'll highlight some of the key new customer awards and cover recent developments across the business. Joe will then take you through our detailed financial results for the third quarter as well as our outlook for the fourth quarter.Our strong third quarter results reflect our ability to consistently drive sustained outperformance. We delivered 11% revenue growth. That represents a record 13 points over market; the result of double-digit growth over market in both our Advanced Safety and User Experience, and Signal and Power Solutions segments.Operating income totaled $420 million, that's up 7%, while earnings per share reached $1.24, an increase of 8% over the prior year. Operating margins declined 40 basis points, driven by unfavorable FX rates and commodity prices. Excluding the impact of FX and commodities, margins increased 30 basis points.In the face of softening global vehicle production, achieving the revenue, operating income and earnings guidance we provided back in July reflects the strong demand for our portfolio of technologies aligned to the safe, green and connected mega trends, as well as our flexible cost structure.We believe it's prudent, given both the weakening of customer schedules that began late in the third quarter and the significant change in FX rates to adjust our financial outlook for the fourth quarter to reflect a more choppy macro environment. Joe will take you through the details in a moment, but we now expect global vehicle production to be down roughly 0.5 point for the full year.Moving to the right side of the slide, we continued our record pace of new business awards totaling almost $16 billion year-to-date, and putting us solidly on track to exceed our prior year record of over $19 billion.Our recent customer awards are at the intersection of Auto 2.0 trends. As always, look to accelerate their adoption of higher levels of advanced safety, electrification and connectivity. And as a result, we're booking new business, because we have the right software, compute and integration capabilities required to help accelerate OE adoption.Lastly, our Mobility and Services Group continues to make progress on next generation automated driving software, vehicle architecture and connected services, which are gaining commercial momentum and will be on display at CES 2019.In summary, another strong quarter, further validating that our operating model, technology portfolio and business strategy can deliver continued outperformance in any environment.On Slide 4, you can see the third quarter new business bookings totaled $4.4 billion, bringing the year-to-date total to $15.6 billion. The record bookings levels are the direct result of our widening competitive mode in several advanced technologies.We booked $2.4 billion of active safety awards year-to-date, and are on track to reach over $3 billion for the full year. Year-to-date infotainment and user experience customer awards totaled $1.5 billion, already surpassing last year's levels.Our engineered components business has booked $4.7 billion of new customer awards year-to-date, including almost $750 million of high voltage connectors, contributing to the 50% [ph] growth in high voltage electrification awards year-to-date.Our continued momentum in new business bookings reinforces our outlook for continued strong revenue growth, driven by our transition to a more integrated solutions provider, creating the software and hardware foundation that enables new features and functions while optimizing the total system cost for the vehicle.Turning to segment highlights in Advanced Safety and User Experience on Slide 5. Sales were up 14%, or 15 points over market, driven by 68% growth in active safety. As we highlighted on last quarter's earnings call, we're starting to lap new infotainment program launches and are gearing up for our next generation Integrated Cockpit Controller launch in 2020.",   
                 ...,
                 'ZTS': "The presentation materials and additional financial tables are currently posted on the Investor Relations section of zoetis.com. The presentation slides can be managed by you, the viewer, and will not be forwarded automatically. In addition, a replay of this call will be available approximately two hours after the conclusion of this call via dial-in or on the Investor Relations section of zoetis.com.At this time, all participants have been placed in a listen-only mode, and the floor will be open for your questions following the presentationIt's now my pleasure to turn the floor over to Steve Frank. Steve, you may begin.Steve Frank - Zoetis, Inc.Thank you. Good morning, everyone, and welcome to the Zoetis Third Quarter 2018 Earnings Call. I am joined today by Juan Ramón Alaix, our Chief Executive Officer, and Glenn David, our Chief Financial Officer. Before we begin, I'll remind you that the slides presented on this call are available on the Investor Relations section of our website and that our remarks today will include forward-looking statements and that actual results could differ materially from those projections. For a list and description of certain factors that could cause results to differ, I refer you to the forward-looking statements, today's press release and our SEC filings including, but not limited to, our Annual Report on Form 10-K and our reports on Form 10-Q.Our remarks today will also include references to certain financial measures, which were not prepared in accordance with generally accepted accounting principles or U.S. GAAP. A reconciliation of these non-GAAP financial measures to the most directly comparable U.S. GAAP measures is included in the financial tables that accompany our earnings press release and in the company's 8-K filing dated today, November 1, 2018. We also cite operational results which exclude the impact of foreign exchange.With that, I will turn the call over to Juan Ramón.Juan Ramón Alaix - Zoetis, Inc.Thank you, Steve. Good morning, everyone. In 2018, we have been making significant effort to support future growth with investment in the business. We are allocating more resources to critical R&D projects, updating and expanding many of our manufacturing capabilities and investing in key areas such as diagnostics with the recent acquisition of Abaxis. We are making good progress integrating Abaxis. Our sales teams have been cross-trained on Zoetis' and Abaxis' product lines in the U.S. and Canada, and they're already collaborating on customer accounts. Outside of the U.S. we are focused on building a more robust diagnostic testing and using the existing Zoetis infrastructure to expand our diagnostics sales.The initial feedback on the acquisition has been very good. Our field force has been very positive about the potential of this combination and the more comprehensive solutions they can now bring to our veterinary customers. Early customer response has been very encouraging. Our core business and investment strategy has Zoetis well-positioned for the future with a diverse portfolio addressing needs across the continuum of care and a promising pipeline of new products and lifecycle innovations.As I mentioned last quarter, we have taken a phased filing (4:38) approach for a new three-way combination parasiticide, and we continue making that progress. The product will be known by the trade name Simparica Trio, and it will combine sarolaner and two other active ingredients to focus on ectoparasites such as fleas, ticks and mites as well as internal parasites and the prevention of heartworm disease.Our Pfizer (5:10) regulatory filings are progressing in the U.S. The safety, efficacy and chemistry manufacturing and control chemical (5:20) sections are under FDA review. In addition, we have submitted the dossier with the European Medicines Agency and subject to regulatory reviews and approval, we would anticipate Simparica Trio coming to market in 2020. We are also progressing well with our monoclonal antibodies program, targeting pain treatment for dogs and cats. This will strengthen our leadership position for pain in dogs and allow us to expand into cats. I will provide further updates as we move through the regulatory process.Our R&D team also continues to enhance our internal portfolio (6:10) with lifecycle innovations and additional (6:14) such as the recent expansion of Cytopoint to cover allergic as well as atopic dermatitis in the U.S. also introducing (6:28) into new geographies and new combinations.Now turning to a few highlights of our third quarter results. Zoetis continued delivering strong revenue growth with 12% operational growth this quarter. Our diverse portfolio remains (6:52), reliable performance with the growth (7:00) across our major species, markets and therapeutic areas. Our companion animal products performed very well in the third quarter based on continued growth in our key dermatology brands, new parasiticides and, for the first time, the addition of Abaxis diagnostic portfolio."})



### 1.3 Scraping announcement dates of transcripts

I forgot to scrape dates on which each earning transcript was published. The majority of code below is, in fact, redundant. I was thinking about merging the two. However, I speculated that I might use one or the other in the future. i.e. pure NLP. Thus, I did not merge it with the code above


```
dates = defaultdict(str)
cnt=0
captcha=[]
captcha2=[]

for symbol in list(my_dict.keys())[:-2]:
    url_path = f'https://www.nasdaq.com/symbol/{symbol}/call-transcripts'
    response = requests.get(url_path)
    soup = BeautifulSoup(response.text,'lxml')
    query = re.compile('Q3 2018')
    if not soup.find(text=query):
        alter_url=f'https://seekingalpha.com/symbol/{symbol}/earnings/transcripts'
        response = requests.get(alter_url)
        soup = BeautifulSoup(response.text,'lxml')
        aug_path = soup.find(text=query).find_parent()['href']
        call_url = f'https://seekingalpha.com'+aug_path
        call_response = requests.get(call_url)
        soup = BeautifulSoup(call_response.text,'lxml')
        raw_transcript = [item.text for item in soup.find(text=query).find_parent().find_parent().find_parent().find_all('p')]
        raw_transcript=raw_transcript[raw_transcript.index('')+1:]
        raw_transcript=''.join(raw_transcript)
    else:
        unrefined_string = soup.find(text=query).find_parents('span')
        aug_path = unrefined_string[0].find('a')['href']
        call_url = f'http://www.nasdaq.com'+aug_path

        call_response = requests.get(call_url)
        soup = BeautifulSoup(call_response.text,'lxml')

        raw_transcript = [item.text for item in soup.find(text=query).find_parent().find_parent().find_parent().find_all('p')]
        raw_transcript=''.join(raw_transcript)
        Months = ['January', 'February','March', 'April','May','June','July','August','September','October','November','December']
        month_cand = []
        for month in Months:
            idx = raw_transcript.find(month)
            if idx > 0:
                month_cand.append(idx)
        idx = min(month_cand)
        idx2 = raw_transcript[idx:].find(' ')
        idx3 = raw_transcript[idx+idx2 +1:].find(' ')
        idx4 = raw_transcript[idx+idx2+idx3 +2:].find(' ')
        exec("dates[f'{symbol}'] = raw_transcript[idx:idx+idx2+idx3+idx4+2]")
        cnt+=1
        if cnt%10 ==0: print(f"{cnt} companies' transcript read")

```

    480 companies' transcript read



```
dates
```




    defaultdict(str,
                {'MMM': 'October 23, 2018',
                 'ABT': 'October 17, 2018',
                 'ABBV': 'November 02, 2018',
                 'ABMD': 'February 01, 2018',
                 'ACN': 'June 28 2018',
                 'ZION': 'October 22, 2018',
                 'ZTS': 'November 01, 2018'})




```
earnings_dates =pd.to_datetime(pd.Series(dates))
earnings_dates.head()
```




    MMM    2018-10-23
    ABT    2018-10-17
    ABBV   2018-11-02
    ABMD   2018-02-01
    ACN    2018-06-28
    dtype: datetime64[ns]



## 2. Create dataframe for training and test

#### target is 'return', which is 1-day holding period return post announcement. 'transcript' will be fed into an NLP model, and the rest will be regressors. All numerical variables were available before the announcement of earnings trancripts.

### 2.1 dataframe for transcripts


```
my_dict_df = pd.DataFrame(my_dict.keys()).rename(columns = {0:'Symbol'})
my_dict_df['transcript'] = my_dict.values()
my_dict_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>transcript</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>I would now like to turn the call over to Bruc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABT</td>
      <td>With the exception of any participant's questi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBV</td>
      <td>I would now like to introduce Ms. Liz Shea, Vi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABMD</td>
      <td>I would now like to introduce your host for to...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>I would now like to turn the conference over t...</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 dataframe for 1-day holding period return and financial variables

Import stock price data from IEX API


```
url = f'https://api.iextrading.com/1.0/stock/{list(my_dict.keys())[:-2][0]}/chart/1y'
price_df = pd.read_json(url)[['date','close']].rename(columns = {'close':list(my_dict.keys())[:-2][0]})
for symbol in list(my_dict.keys())[1:-2]:
    url= f'https://api.iextrading.com/1.0/stock/{symbol}/chart/1y'
    price_df = price_df.merge(pd.read_json(url)[['date','close']].rename(columns = {'close':symbol}), on='date',how='outer')
price_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>MMM</th>
      <th>ABT</th>
      <th>ABBV</th>
      <th>ABMD</th>
      <th>ACN</th>
      <th>ATVI</th>
      <th>ADBE</th>
      <th>AMD</th>
      <th>AAP</th>
      <th>...</th>
      <th>WLTW</th>
      <th>WYNN</th>
      <th>XEL</th>
      <th>XRX</th>
      <th>XLNX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-23</td>
      <td>210.0888</td>
      <td>57.8309</td>
      <td>88.9667</td>
      <td>302.65</td>
      <td>150.4081</td>
      <td>65.7022</td>
      <td>225.30</td>
      <td>10.04</td>
      <td>105.3581</td>
      <td>...</td>
      <td>148.6723</td>
      <td>188.9819</td>
      <td>43.8960</td>
      <td>29.9704</td>
      <td>62.5995</td>
      <td>77.5583</td>
      <td>84.4235</td>
      <td>110.3229</td>
      <td>53.0850</td>
      <td>84.7505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-24</td>
      <td>195.7345</td>
      <td>57.3099</td>
      <td>87.2384</td>
      <td>290.93</td>
      <td>148.4720</td>
      <td>64.8193</td>
      <td>217.89</td>
      <td>10.09</td>
      <td>106.7660</td>
      <td>...</td>
      <td>147.2649</td>
      <td>185.8133</td>
      <td>44.2545</td>
      <td>29.4901</td>
      <td>62.9139</td>
      <td>74.5749</td>
      <td>83.7256</td>
      <td>108.5377</td>
      <td>53.8971</td>
      <td>83.8067</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-25</td>
      <td>193.5059</td>
      <td>57.6441</td>
      <td>87.7254</td>
      <td>293.20</td>
      <td>147.5088</td>
      <td>64.3034</td>
      <td>217.32</td>
      <td>9.71</td>
      <td>107.6546</td>
      <td>...</td>
      <td>146.9893</td>
      <td>178.8600</td>
      <td>44.3417</td>
      <td>29.4325</td>
      <td>62.5307</td>
      <td>74.6145</td>
      <td>83.8042</td>
      <td>109.6782</td>
      <td>54.0634</td>
      <td>84.0252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-26</td>
      <td>191.8029</td>
      <td>58.4795</td>
      <td>93.0441</td>
      <td>303.41</td>
      <td>149.6710</td>
      <td>65.8113</td>
      <td>221.91</td>
      <td>11.04</td>
      <td>113.0064</td>
      <td>...</td>
      <td>147.5798</td>
      <td>180.1998</td>
      <td>45.0295</td>
      <td>29.2211</td>
      <td>63.5919</td>
      <td>73.8242</td>
      <td>85.2786</td>
      <td>114.1511</td>
      <td>53.8775</td>
      <td>84.1842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-27</td>
      <td>190.8589</td>
      <td>58.5385</td>
      <td>94.2759</td>
      <td>301.74</td>
      <td>149.7693</td>
      <td>65.2657</td>
      <td>221.90</td>
      <td>11.11</td>
      <td>116.1615</td>
      <td>...</td>
      <td>148.3672</td>
      <td>181.0310</td>
      <td>45.6302</td>
      <td>30.0473</td>
      <td>63.6312</td>
      <td>72.8165</td>
      <td>85.5637</td>
      <td>115.9561</td>
      <td>54.3960</td>
      <td>84.2935</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 482 columns</p>
</div>



date column is not in datetime object. But, I want to obtain 1-day holding period return on stock from the transcript announcement date. This requires timedelta object. Thus, I change date column to be a datetime object


```
price_df.loc[:,'date'] = pd.to_datetime(price_df['date'])
price_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>MMM</th>
      <th>ABT</th>
      <th>ABBV</th>
      <th>ABMD</th>
      <th>ACN</th>
      <th>ATVI</th>
      <th>ADBE</th>
      <th>AMD</th>
      <th>AAP</th>
      <th>...</th>
      <th>WLTW</th>
      <th>WYNN</th>
      <th>XEL</th>
      <th>XRX</th>
      <th>XLNX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-23</td>
      <td>210.0888</td>
      <td>57.8309</td>
      <td>88.9667</td>
      <td>302.65</td>
      <td>150.4081</td>
      <td>65.7022</td>
      <td>225.30</td>
      <td>10.04</td>
      <td>105.3581</td>
      <td>...</td>
      <td>148.6723</td>
      <td>188.9819</td>
      <td>43.8960</td>
      <td>29.9704</td>
      <td>62.5995</td>
      <td>77.5583</td>
      <td>84.4235</td>
      <td>110.3229</td>
      <td>53.0850</td>
      <td>84.7505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-24</td>
      <td>195.7345</td>
      <td>57.3099</td>
      <td>87.2384</td>
      <td>290.93</td>
      <td>148.4720</td>
      <td>64.8193</td>
      <td>217.89</td>
      <td>10.09</td>
      <td>106.7660</td>
      <td>...</td>
      <td>147.2649</td>
      <td>185.8133</td>
      <td>44.2545</td>
      <td>29.4901</td>
      <td>62.9139</td>
      <td>74.5749</td>
      <td>83.7256</td>
      <td>108.5377</td>
      <td>53.8971</td>
      <td>83.8067</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-25</td>
      <td>193.5059</td>
      <td>57.6441</td>
      <td>87.7254</td>
      <td>293.20</td>
      <td>147.5088</td>
      <td>64.3034</td>
      <td>217.32</td>
      <td>9.71</td>
      <td>107.6546</td>
      <td>...</td>
      <td>146.9893</td>
      <td>178.8600</td>
      <td>44.3417</td>
      <td>29.4325</td>
      <td>62.5307</td>
      <td>74.6145</td>
      <td>83.8042</td>
      <td>109.6782</td>
      <td>54.0634</td>
      <td>84.0252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-26</td>
      <td>191.8029</td>
      <td>58.4795</td>
      <td>93.0441</td>
      <td>303.41</td>
      <td>149.6710</td>
      <td>65.8113</td>
      <td>221.91</td>
      <td>11.04</td>
      <td>113.0064</td>
      <td>...</td>
      <td>147.5798</td>
      <td>180.1998</td>
      <td>45.0295</td>
      <td>29.2211</td>
      <td>63.5919</td>
      <td>73.8242</td>
      <td>85.2786</td>
      <td>114.1511</td>
      <td>53.8775</td>
      <td>84.1842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-27</td>
      <td>190.8589</td>
      <td>58.5385</td>
      <td>94.2759</td>
      <td>301.74</td>
      <td>149.7693</td>
      <td>65.2657</td>
      <td>221.90</td>
      <td>11.11</td>
      <td>116.1615</td>
      <td>...</td>
      <td>148.3672</td>
      <td>181.0310</td>
      <td>45.6302</td>
      <td>30.0473</td>
      <td>63.6312</td>
      <td>72.8165</td>
      <td>85.5637</td>
      <td>115.9561</td>
      <td>54.3960</td>
      <td>84.2935</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 482 columns</p>
</div>




```
filt = (price_df['date'] - earnings_dates[SYMBOLS[0]]>pd.Timedelta(0))
for symbol in price_df.columns[2:]:
    if my_dict[symbol] != '':
        filt = pd.concat([filt,(price_df['date'] - earnings_dates[symbol]>pd.Timedelta(0))], axis=1)
    else:
        filt[symbol] = False
filt.columns = price_df.columns[1:]
```


```
filt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMM</th>
      <th>ABT</th>
      <th>ABBV</th>
      <th>ABMD</th>
      <th>ACN</th>
      <th>ATVI</th>
      <th>ADBE</th>
      <th>AMD</th>
      <th>AAP</th>
      <th>AES</th>
      <th>...</th>
      <th>WLTW</th>
      <th>WYNN</th>
      <th>XEL</th>
      <th>XRX</th>
      <th>XLNX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 481 columns</p>
</div>



Financial Indicators


```
url = f'https://api.iextrading.com/1.0/stock/{list(my_dict.keys())[0]}/stats'
other_df = pd.read_json(url,lines=True)
for symbol in list(my_dict.keys())[1:-2]:
    url= f'https://api.iextrading.com/1.0/stock/{symbol}/stats'
    other_df = pd.concat([other_df,pd.read_json(url,lines=True)])
other_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EBITDA</th>
      <th>EPSSurpriseDollar</th>
      <th>EPSSurprisePercent</th>
      <th>beta</th>
      <th>cash</th>
      <th>companyName</th>
      <th>consensusEPS</th>
      <th>day200MovingAvg</th>
      <th>day30ChangePercent</th>
      <th>day50MovingAvg</th>
      <th>...</th>
      <th>shortRatio</th>
      <th>symbol</th>
      <th>ttmEPS</th>
      <th>week52change</th>
      <th>week52high</th>
      <th>week52low</th>
      <th>year1ChangePercent</th>
      <th>year2ChangePercent</th>
      <th>year5ChangePercent</th>
      <th>ytdChangePercent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>1.058601</td>
      <td>0</td>
      <td>3M Company</td>
      <td>2.50</td>
      <td>201.23337</td>
      <td>0.066569</td>
      <td>209.45845</td>
      <td>...</td>
      <td>4.44</td>
      <td>MMM</td>
      <td>9.98</td>
      <td>4.4796</td>
      <td>219.67</td>
      <td>176.87</td>
      <td>0.121417</td>
      <td>0.187799</td>
      <td>0.822215</td>
      <td>0.157463</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>1.098760</td>
      <td>0</td>
      <td>Abbott Laboratories</td>
      <td>0.61</td>
      <td>70.15548</td>
      <td>-0.023371</td>
      <td>77.22746</td>
      <td>...</td>
      <td>2.56</td>
      <td>ABT</td>
      <td>2.88</td>
      <td>31.5907</td>
      <td>80.74</td>
      <td>56.81</td>
      <td>0.327868</td>
      <td>0.789935</td>
      <td>1.202490</td>
      <td>0.104544</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>1.143775</td>
      <td>0</td>
      <td>AbbVie Inc.</td>
      <td>2.05</td>
      <td>85.50070</td>
      <td>0.012229</td>
      <td>79.34659</td>
      <td>...</td>
      <td>3.75</td>
      <td>ABBV</td>
      <td>7.91</td>
      <td>-11.5849</td>
      <td>107.25</td>
      <td>75.77</td>
      <td>-0.098333</td>
      <td>0.317200</td>
      <td>0.918742</td>
      <td>-0.095815</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>-2.5000</td>
      <td>0.264236</td>
      <td>0</td>
      <td>ABIOMED Inc.</td>
      <td>0.80</td>
      <td>351.86725</td>
      <td>-0.190203</td>
      <td>313.57960</td>
      <td>...</td>
      <td>NaN</td>
      <td>ABMD</td>
      <td>2.72</td>
      <td>-13.3719</td>
      <td>459.75</td>
      <td>244.08</td>
      <td>-0.098821</td>
      <td>1.028472</td>
      <td>9.453748</td>
      <td>-0.154149</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>6.5217</td>
      <td>1.115795</td>
      <td>0</td>
      <td>Accenture plc Class A (Ireland)</td>
      <td>1.84</td>
      <td>160.34926</td>
      <td>0.109671</td>
      <td>166.95319</td>
      <td>...</td>
      <td>2.20</td>
      <td>ACN</td>
      <td>6.91</td>
      <td>19.9936</td>
      <td>180.55</td>
      <td>132.63</td>
      <td>0.215583</td>
      <td>0.567132</td>
      <td>1.548005</td>
      <td>0.294366</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```

other_lookup_df=other_df[['symbol','returnOnEquity','ttmEPS','returnOnAssets']]
other_lookup_df = other_lookup_df.reset_index(drop=True)
print(other_lookup_df.shape)
other_lookup_df.tail()
```

    (481, 4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>returnOnEquity</th>
      <th>ttmEPS</th>
      <th>returnOnAssets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>476</th>
      <td>XYL</td>
      <td>20.83</td>
      <td>2.89</td>
      <td>7.80</td>
    </tr>
    <tr>
      <th>477</th>
      <td>YUM</td>
      <td>-22.05</td>
      <td>3.16</td>
      <td>33.47</td>
    </tr>
    <tr>
      <th>478</th>
      <td>ZBH</td>
      <td>-3.31</td>
      <td>7.64</td>
      <td>-1.52</td>
    </tr>
    <tr>
      <th>479</th>
      <td>ZION</td>
      <td>12.20</td>
      <td>4.10</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>480</th>
      <td>ZTS</td>
      <td>72.21</td>
      <td>3.14</td>
      <td>14.75</td>
    </tr>
  </tbody>
</table>
</div>




```
X=pd.concat([other_lookup_df,my_dict_df['transcript'][:-2]],axis=1)
X.head()
X.keys()[:-1]
```


```
X.values.reshape(-1,6)
```


```
# filter for post transcript stock prices
filt = (price_df['date'] - earnings_dates[price_df.columns[1]]>pd.Timedelta(0))
for symbol in price_df.columns[2:]:
    if my_dict[symbol] != '':
        filt = pd.concat([filt,(price_df['date'] - earnings_dates[symbol]>pd.Timedelta(0))], axis=1)
    else:
        filt[symbol] = False
filt.columns = price_df.columns[1:]
price_cumul = price_df.loc[:,'MMM':'ZTS']
price_cumul=price_cumul[filt]
price_cumul.index = price_df['date']
```








```
# 1-day holding period return from the announcement date
r= []
for symbol in price_df.columns[1:]:
    if price_cumul[symbol].notna().any():
        r+=[(price_cumul[symbol][price_cumul[symbol].notna()][1]/price_cumul[symbol]\
        [price_cumul[symbol].notna()][0] -1)*100]
r[:5]
```




    [0.24386116998427187,
     -0.8071177959703468,
     1.3077553656597685,
     -3.8724599372212065,
     -0.2200568988273255]




```
data_dropna = X.copy()
data_dropna['return'] = r
data_dropna.dropna(inplace=True)
```


```
train,test= train_test_split(data_dropna,test_size=0.33,random_state=42)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>returnOnEquity</th>
      <th>ttmEPS</th>
      <th>returnOnAssets</th>
      <th>transcript</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>TSS</td>
      <td>24.36</td>
      <td>4.27</td>
      <td>8.52</td>
      <td>I would now like to turn the conference over t...</td>
      <td>2.855239</td>
    </tr>
    <tr>
      <th>211</th>
      <td>GWW</td>
      <td>42.55</td>
      <td>16.70</td>
      <td>13.36</td>
      <td>I’d now like to turn the conference over to yo...</td>
      <td>-1.578707</td>
    </tr>
    <tr>
      <th>118</th>
      <td>CMCSA</td>
      <td>16.73</td>
      <td>2.56</td>
      <td>5.34</td>
      <td>I will now turn the call over to Senior Vice P...</td>
      <td>2.128361</td>
    </tr>
    <tr>
      <th>242</th>
      <td>ICE</td>
      <td>11.89</td>
      <td>3.59</td>
      <td>2.37</td>
      <td>I would now like to turn the conference over t...</td>
      <td>1.384153</td>
    </tr>
    <tr>
      <th>160</th>
      <td>EIX</td>
      <td>-5.79</td>
      <td>4.14</td>
      <td>-1.17</td>
      <td>Sam Ramraj - Edison InternationalThank you, As...</td>
      <td>-1.729342</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Implement models

I am going to put every feature and target into pipelines. This will enable me to crossvalidate every component.
For example, ItemSelector defined below will separate financial variables from transcripts. Then, transcripts will be analyzed by an NLP model called latent dirichlet allocation. The financial variables and outcome of the NLP model will be merged via featureunion. This feature union will be regressed by the target variable, return.


```
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

```

Below is an example output of the Pipeline for turning actual transcript to latent topics. Rows correspond to distribution per transcript. But the first five sample of preliminary result is not satisfactory to me because distributions are rather homogeneous.


```
word = Pipeline([
                ('selector', ItemSelector(key='transcript')),
                ('vect', CountVectorizer()),
                ('lda',LatentDirichletAllocation(n_components=5))
])
pd.DataFrame(word.fit_transform(train), columns=topic_labels).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic1</th>
      <th>Topic2</th>
      <th>Topic3</th>
      <th>Topic4</th>
      <th>Topic5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000256</td>
      <td>0.998973</td>
      <td>0.000255</td>
      <td>0.000255</td>
      <td>0.000261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000262</td>
      <td>0.998950</td>
      <td>0.000260</td>
      <td>0.000261</td>
      <td>0.000267</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000189</td>
      <td>0.786655</td>
      <td>0.000188</td>
      <td>0.000188</td>
      <td>0.212780</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.062572</td>
      <td>0.936610</td>
      <td>0.000271</td>
      <td>0.000270</td>
      <td>0.000277</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000224</td>
      <td>0.000227</td>
      <td>0.000222</td>
      <td>0.000222</td>
      <td>0.999105</td>
    </tr>
  </tbody>
</table>
</div>



Below is an example output of the Pipeline for turning actual transcript to latent topics. Rows correspond to distribution per transcript. But the first five sample of preliminary result is not satisfactory to me because distributions are rather homogeneous.


```
finance = Pipeline([
                ('selector', ItemSelector(key=train.keys()[1:-2])),
                ('scaler', StandardScaler()),
])
```




    array([[-1.01014747e-02, -2.97361780e-01,  1.85635798e-01],
           [ 8.43596733e-02,  1.98330889e+00,  8.52854913e-01],
           [-4.97242652e-02, -6.11114543e-01, -2.52743704e-01],
           [-7.48585124e-02, -4.22128961e-01, -6.62173615e-01],
           [-1.66671217e-01, -3.21214329e-01, -1.15018098e+00],
           ...,
           [-5.09705916e-02, -4.60660002e-01, -3.60270751e-01],
           [-1.27065667e+00, -7.28542478e-01,  3.27626642e-01],
           [ 2.59416589e-01, -2.60665551e-01,  4.93052869e-01]])




```
feature = FeatureUnion(
         transformer_list = [
             ('lda', word),
             ('finance',finance)
        ],

        transformer_weights = {
        'lda': 0.8,
        'finance': 1,
        },
)
print(feature.fit_transform(train).shape)
print(feature.fit_transform(train))

```

    (300, 8)
    [[ 2.06227587e-04  7.99180228e-01  2.06677176e-04 ... -1.01014747e-02
      -2.97361780e-01  1.85635798e-01]
     [ 2.09879131e-04  7.99163717e-01  2.11565317e-04 ...  8.43596733e-02
       1.98330889e+00  8.52854913e-01]
     [ 1.50615289e-04  7.99397620e-01  1.52941789e-04 ... -4.97242652e-02
      -6.11114543e-01 -2.52743704e-01]
     ...
     [ 2.23534816e-04  7.99108085e-01  2.25342441e-04 ... -5.09705916e-02
      -4.60660002e-01 -3.60270751e-01]
     [ 2.12040114e-04  7.99147380e-01  2.14957516e-04 ... -1.27065667e+00
      -7.28542478e-01  3.27626642e-01]
     [ 1.92102013e-04  7.02455041e-01  9.69729748e-02 ...  2.59416589e-01
      -2.60665551e-01  4.93052869e-01]]



```
model = Pipeline([('feature',feature),('ridge', Ridge())])
base_model = model.fit(train,train['return'])
base_model.score(test,test['return'])
```




    -0.05262336593522554




```
model.get_params().keys()
```




    dict_keys(['memory', 'steps', 'feature', 'ridge', 'feature__n_jobs', 'feature__transformer_list', 'feature__transformer_weights', 'feature__lda', 'feature__finance', 'feature__lda__memory', 'feature__lda__steps', 'feature__lda__selector', 'feature__lda__vect', 'feature__lda__lda', 'feature__lda__selector__key', 'feature__lda__vect__analyzer', 'feature__lda__vect__binary', 'feature__lda__vect__decode_error', 'feature__lda__vect__dtype', 'feature__lda__vect__encoding', 'feature__lda__vect__input', 'feature__lda__vect__lowercase', 'feature__lda__vect__max_df', 'feature__lda__vect__max_features', 'feature__lda__vect__min_df', 'feature__lda__vect__ngram_range', 'feature__lda__vect__preprocessor', 'feature__lda__vect__stop_words', 'feature__lda__vect__strip_accents', 'feature__lda__vect__token_pattern', 'feature__lda__vect__tokenizer', 'feature__lda__vect__vocabulary', 'feature__lda__lda__batch_size', 'feature__lda__lda__doc_topic_prior', 'feature__lda__lda__evaluate_every', 'feature__lda__lda__learning_decay', 'feature__lda__lda__learning_method', 'feature__lda__lda__learning_offset', 'feature__lda__lda__max_doc_update_iter', 'feature__lda__lda__max_iter', 'feature__lda__lda__mean_change_tol', 'feature__lda__lda__n_components', 'feature__lda__lda__n_jobs', 'feature__lda__lda__n_topics', 'feature__lda__lda__perp_tol', 'feature__lda__lda__random_state', 'feature__lda__lda__topic_word_prior', 'feature__lda__lda__total_samples', 'feature__lda__lda__verbose', 'feature__finance__memory', 'feature__finance__steps', 'feature__finance__selector', 'feature__finance__scaler', 'feature__finance__selector__key', 'feature__finance__scaler__copy', 'feature__finance__scaler__with_mean', 'feature__finance__scaler__with_std', 'ridge__alpha', 'ridge__copy_X', 'ridge__fit_intercept', 'ridge__max_iter', 'ridge__normalize', 'ridge__random_state', 'ridge__solver', 'ridge__tol'])



param_grid = {
    'feature__lda__vect__max_df':[0.5,0.6,0.8,0.9,1.0],
    'feature__lda__vect__min_df':[0.01,0.05,0.1],
    'feature__lda__lda__max_iter':[500,1000],
    'feature__lda__lda__n_components':[3,4,5,6,7,8,9,10,11],
    'ridge__alpha': [0.01,0.1,1,5,10],
}
grid = GridSearchCV(model, cv=5, param_grid=param_grid)
grid.fit(train,train['return'])


```
hyperparameters = { 'feature__lda__vect__max_df': [0.5, 0.6],
                    'feature__lda__vect__min_df': [0.01,0.05],
                    'feature__lda__vect__ngram_range': [(1,1), (1,2)],
                    'feature__lda__lda__n_components': [3,5,7,11],
                    'ridge__alpha': [0.01,0.1,1,5,10],
                  }
clf = GridSearchCV(model, hyperparameters, cv=5)

# Fit and tune model
clf.fit(train,train['return'])
```




    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=Pipeline(memory=None,
         steps=[('feature', FeatureUnion(n_jobs=None,
           transformer_list=[('lda', Pipeline(memory=None,
         steps=[('selector', ItemSelector(key='transcript')), ('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='c...it_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001))]),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'feature__lda__vect__max_df': [0.5, 0.6], 'feature__lda__vect__min_df': [0.01, 0.05], 'feature__lda__vect__ngram_range': [(1, 1), (1, 2)], 'feature__lda__lda__n_components': [3, 5, 7, 11], 'ridge__alpha': [0.01, 0.1, 1, 5, 10]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```
clf.best_params_
```




    {'feature__lda__lda__n_components': 11,
     'feature__lda__vect__max_df': 0.5,
     'feature__lda__vect__min_df': 0.05,
     'feature__lda__vect__ngram_range': (1, 2),
     'ridge__alpha': 0.01}




```
clf.refit

clf.score(test,test['return'])

```




    -0.05253008864787767



In the presentation slide, I incorporated many plots and word graphics. This notebook is created after the slide was made and is meant for summary of what had been done. Currently, I am interested in exploring other models. Thus, I will come back to graphics if I have more freetime !!! Thank you.
