import numpy as np
import cv2
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By


def get_image_from_url(url):
    try:
        # Open the WebScore AI website
        driver.get(url)
    except Exception as e:
        print(e)
        return None

    # Give time for website to load
    time.sleep(5)

    # Capture the screenshot as binary data
    try:
        # Capture the full-length screenshot
        screenshot = driver.find_element(By.TAG_NAME, 'body').screenshot_as_png
    except Exception as e:
        print(e)
        return None

    # Convert the binary data to a NumPy array
    screenshot_array = np.asarray(bytearray(screenshot), dtype=np.uint8)

    # Read the array as a cv2 image
    image = cv2.imdecode(screenshot_array, cv2.IMREAD_COLOR)

    # Convert the cv2 image from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    

def close_driver():
    driver.quit()


options = uc.ChromeOptions() 
options.headless = True
options.add_argument('--ignore-ssl-errors=yes')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--start-maximized')  # Maximize the browser window
options.add_argument('--hide-scrollbars')  # Hide the scrollbars
driver = uc.Chrome(use_subprocess=True, options=options) 