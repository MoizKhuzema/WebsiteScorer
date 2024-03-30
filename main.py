import model_1
import model_2
import company_analysis
import screenshot
import pandas as pd
import cv2


def main(filename):
    # read input file
    input = pd.read_excel(filename)
    url_list = input.iloc[:, 0].tolist()

    df = pd.DataFrame(columns=['Url', 'Name', 'Description', 'Country', 'Industry', 
                               'Address', 'MarketCapitalization', 'DividendPerShare', 
                                'ProfitMargin', 'RevenueTTM', 'GrossProfitTTM', 'Operator', 
                                'Managing director','Telephone no.', 'Email', 'Servers',  
                                'Technology Stack', 'Webscore (0-9)', 'Webscore (0-99)', 'Remarks'])
    i = 1
    for url in url_list:
        print(i)
        # Company Analysis
        try:
            tech = company_analysis.find_tech_stack(url)
        except Exception as e:
            print(e)
            tech = None

        try:
            info = company_analysis.find_operator_info(url)
        except Exception as e:
            print(e)
            info = None
        try: 
            analysis = company_analysis.find_company_analysis(url)
        except Exception as e:
            print(e)
            analysis = None

        # Website scoring
        image = screenshot.get_image_from_url(url)
        if image is None:
            score_1 = 'NaN'
            score_1 = 'NaN'
            remarks = 'NaN'
        else:
            score_1 = model_1.predict(image)
            score_2, remarks = model_2.predict(image)

        # Write result
        res = [url]

        if analysis == None:
            res.extend(['NaN','NaN','NaN','NaN','NaN','NaN','NaN','NaN','NaN','NaN'])
        else:
            analysis = ['NaN' if item is None else item for item in analysis]
            res.extend(analysis)

        if info == None:
            res.extend(['NaN','NaN','NaN','NaN','NaN'])
        else:
            info = ['NaN' if item is None else item for item in info]
            res.extend(info)

        if tech == None:
            res.extend(['NaN'])
        else:
            res.extend([', '.join(tech)])

        res.extend([score_1, score_2, remarks])
            
        df.loc[len(df.index)] = res
        i += 1

    screenshot.close_driver()
    # Save the DataFrame as an Excel file
    df.to_excel('output.xlsx', index=False)

main('C:\Users\moizk\Desktop\Upwork\Marcel-M\input\Website-Sample.xlsx')