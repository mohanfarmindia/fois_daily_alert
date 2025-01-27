!pip install -r requirements.txt

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Load data from CSV
df = pd.read_csv('updated_filtered_df.csv')

# Setup Chrome options for headless mode
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--start-maximized")

# Initialize webdriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the URL
url = 'https://www.fois.indianrail.gov.in/FOISWebPortal/pages/FWP_FrgtCalcNew.jsp'
driver.get(url)

# Wait for the page to load and input field to be present
try:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[1]/input')))
except Exception as e:
    print(f"Error loading page: {e}")
    driver.quit()
    exit()

# Initialize an empty list to store all row data
all_rows_data = []

# Loop through the "STTN FROM", "DSTN", "INDENTED TYPE", "RAKE CMDT" values in the CSV and process them
for index, row in df.iterrows():
    try:
        station_from = row['STTN FROM']
        station_to = row['DSTN']
        indented_type = row['INDENTED TYPE']
        rake_cmdt = row['RAKE CMDT']

        # Input "STTN FROM"
        station_from_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[1]/input')
        station_from_input.clear()
        station_from_input.send_keys(station_from)
        station_from_input.send_keys(Keys.ENTER)

        # Input "DSTN"
        station_to_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[2]/input')
        station_to_input.clear()
        station_to_input.send_keys(station_to)
        station_to_input.send_keys(Keys.ENTER)

        # Input "INDENTED TYPE"
        indented_type_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[3]/input')
        indented_type_input.clear()
        indented_type_input.send_keys(indented_type)
        indented_type_input.send_keys(Keys.ENTER)

        # Conditional input for "FOODGRAINS,FLOURS AND PULSES"
        food_grains_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[5]/input')
        food_grains_input.clear()
        if rake_cmdt == "M":
            food_grains_input.send_keys("FOODGRAINS,FLOURS AND PULSES")
        elif rake_cmdt == "DOC":
            food_grains_input.send_keys("OIL CAKES AND SEEDS")
        food_grains_input.send_keys(Keys.ENTER)

        # Conditional input for "MAIZE" or "DE-OILED CAKES"
        maize_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[6]/input')
        maize_input.clear()
        if rake_cmdt == "M":
            maize_input.send_keys("MAIZE  ")
        elif rake_cmdt == "DOC":
            maize_input.send_keys("DE-OILED CAKES")
        maize_input.send_keys(Keys.ENTER)
        
        time.sleep(1)  # Allow time for processing

        # Wait for iframe and switch to it
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/form/div/center/div/div/iframe')))
        
        driver.switch_to.frame(driver.find_element(By.XPATH, '/html/body/div[4]/form/div/center/div/div/iframe'))

        # Wait for table row 14
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/table/tbody/tr[14]')))

        # Extract row 14 data
        row_data = []
        row_xpath = '/html/body/div[1]/table/tbody/tr[14]/td'
        
        row_cells = driver.find_elements(By.XPATH, row_xpath)
        
        if row_cells:
            for cell in row_cells:
                row_data.append(cell.text)
            print(f"Extracted data for row {index + 1}: {row_data}")
            
            # Extract data from row 8 (specific column)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/table/tbody/tr[8]')))
            heading = driver.find_element(By.XPATH, '/html/body/div[1]/table/tbody/tr[8]/td[1]').text
            value = driver.find_element(By.XPATH, '/html/body/div[1]/table/tbody/tr[8]/td[3]').text.replace(",", "")
            
            # Append data
            row_data.append(float(value))
            all_rows_data.append([station_from, station_to] + row_data)
        
            # Switch back to default content after extraction
            driver.switch_to.default_content()
        
        else:
            print(f"No data found for row {index + 1}")
    
    except Exception as e:
        print(f"Error processing row {index + 1}: {e}")
    
# Create DataFrame from collected data
if all_rows_data:
    columns = ['Station From', 'Station To', 'Charge Name', 'Wagon Load', 'Train Load', heading]
    df_result = pd.DataFrame(all_rows_data, columns=columns)

    # Convert columns to numeric
    df_result['Train Load'] = pd.to_numeric(df_result['Train Load'].str.replace(",", ""), errors='coerce')
    df_result[heading] = pd.to_numeric(df_result[heading], errors='coerce')

    # Calculate Per Quintal Charges if valid data exists
    if df_result['Train Load'].isna().any() or df_result[heading].isna().any():
        print("Missing or invalid data detected.")
    else:
        df_result['Per Quintal Charges'] = (df_result['Train Load'] / (df_result[heading] * 10)).round(2)
        print("Per Quintal Charges calculated successfully.")

    # Save to CSV
    df_result.to_csv("updated_daily_indents_freight_info.csv", index=False)
else:
    print("No valid data was collected.")

# Close driver
driver.quit()
