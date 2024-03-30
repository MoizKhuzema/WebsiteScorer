import model_1
import model_2
import screenshot
import pandas as pd
import cv2


def main(filename):
    # read input file
    input = pd.read_excel(filename)
    url_list = input.iloc[:, 0].tolist()

    df = pd.DataFrame(columns=['Url', 'Websoree Filtered', 'Webscore Unfiltered', 'Remarks'])
    i = 1
    for url in url_list:
        print(i)

        # Website scoring
        image = screenshot.get_image_from_url(url)
        if image is None:
            score_1 = 'NaN'
            score_2 = 'NaN'
            remarks = 'NaN'
        else:
            cv2.imwrite(r'C:\Users\moizk\Desktop\Upwork\Marcel-M\output' + f'\{i}.png', image)
            # score_1 = model_1.predict(image)
            score_1 = 'NaN'
            score_2, score_3 = model_2.predict(image)

            if score_2 < 31 and score_3 < 1000:
                remarks = 'bad website'
            elif score_2 == 'NaN' or score_3 == 'NaN':
                remarks = 'error accessing website'
            else:
                remarks = 'good website'

        # Write result
        res = [url, score_2, score_3, remarks]
        df.loc[len(df.index)] = res
        i += 1

    screenshot.close_driver()
    # Save the DataFrame as an Excel file
    df.to_excel(r'C:\Users\moizk\Desktop\Upwork\Marcel-M\output\output.xlsx', index=False)

main(r'C:\Users\moizk\Desktop\Upwork\Marcel-M\input\input.xlsx')