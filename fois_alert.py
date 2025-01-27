# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from io import BytesIO
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\intel\OneDrive\Documents\FarmIndia_python\railway_frieght_charge_scrapper\tesseract.exe"

# Function to solve CAPTCHA with retry logic
def solve_captcha(driver, image_xpath):
    while True:
        try:
            captcha_element = driver.find_element(By.XPATH, image_xpath)
            captcha_image = captcha_element.screenshot_as_png
            captcha_image = Image.open(BytesIO(captcha_image))
            captcha_image = captcha_image.convert("L")
            captcha_image = captcha_image.filter(ImageFilter.MedianFilter(size=3))
            enhancer = ImageEnhance.Contrast(captcha_image)
            captcha_image = enhancer.enhance(2)
            captcha_text = pytesseract.image_to_string(captcha_image, config='--psm 6')
            cleaned_text = re.sub(r'[^A-Za-z0-9]', '', captcha_text)
            captcha_field = driver.find_element(By.XPATH, "//*[@id='captchaText']")
            captcha_field.clear()
            captcha_field.send_keys(cleaned_text)
            submit_button = driver.find_element(By.XPATH, "//*[@id='collapse1']/div[5]/button")
            submit_button.click()
            time.sleep(2)
            if not is_captcha_incorrect(driver, "//*[@id='errmsg']"):
                print("Captcha accepted, proceeding...")
                return True
        except NoSuchElementException:
            print("CAPTCHA element not found. Retrying...")
        except Exception as e:
            print(f"Error while solving CAPTCHA: {e}")
            time.sleep(1)

def is_captcha_incorrect(driver, error_xpath):
    try:
        error_message = driver.find_element(By.XPATH, error_xpath).text
        if "Captcha Code doesn't Match" in error_message:
            print("Detected Captcha error: Code doesn't match. Retrying...")
            return True
    except NoSuchElementException:
        pass
    return False

def process_region(region_code):
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    region_data_list = []
    try:
        url = "https://www.fois.indianrail.gov.in/FOISWebPortal/pages/FWP_ODROtsgDtls.jsp"
        driver.get(url)
        print(f"\nPage Loaded for region: {region_code}")
        wait = WebDriverWait(driver, 120)

        outstanding_odr_option = wait.until(EC.presence_of_element_located((By.ID, "Zone")))
        outstanding_odr_option.click()
        outstanding_odr_option.send_keys(region_code)
        print(f"Selected '{region_code}' from the dropdown.")
        
        captcha_image_xpath = "/html/body/div[4]/center/form/div/div[2]/div[4]/img[1]"
        if not solve_captcha(driver, captcha_image_xpath):
            raise Exception(f"Unable to solve Captcha for region {region_code} after multiple attempts.")

        print("Waiting for iframe to load...")
        data_div = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='dataDiv']")))
        iframe = data_div.find_element(By.TAG_NAME, "iframe")
        driver.switch_to.frame(iframe)

        print("Waiting for the table to load...")
        table_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "body > div > table")))
        tbody_element = table_element.find_element(By.TAG_NAME, "tbody")
        rows = tbody_element.find_elements(By.TAG_NAME, "tr")

        for row in rows:
            columns = [col.text for col in row.find_elements(By.TAG_NAME, "td")]
            
            # Collect all rows without filtering based on conditions.
            if len(columns) >= 10:  # Ensure there are enough columns before appending.
                print(f"Scraped Data: {columns}")  # Print each row's data in real-time.
                region_data_list.append(columns)

        if not region_data_list:
            print(f"No valid data found for region {region_code}. Skipping this region.")
            return None
        
        column_names = [
            "S.No.", "DVSN", "STTN FROM", "DEMAND NO.", "DEMAND DATE", 
            "DEMAND TIME", "Expected loading date", "CNSR", 
            "CNSG", "CMDT", "TT", "PC", 
            "PBF", "VIA", "RAKE CMDT", 
            "DSTN", "INDENTED TYPE", 
            "INDENTED UNTS", "INDENTED 8W", 
            "OTSG UNTS", "OTSG 8W", 
            "SUPPLIED UNTS", "SUPPLIED TIME"
        ]
        
        df_region = pd.DataFrame(region_data_list, columns=column_names)
        df_region['Region'] = region_code
        
        return df_region
    
    except Exception as e:
        print(f"An error occurred while processing region {region_code}: {e}")
        return None
    
    finally:
        driver.quit()

# Load region codes from text file instead of hardcoding them
with open('region_codes.txt', 'r') as file:
    region_codes = [line.strip() for line in file.readlines()]

# Combine all region DataFrames into a single DataFrame using parallel processing
all_regions_data = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_region, region_codes), total=len(region_codes), desc="Processing Regions"))

# Filter out None results and concatenate DataFrames into a single DataFrame and save it.
results_filtered = [df for df in results if df is not None]
if results_filtered:
    final_df = pd.concat(results_filtered, ignore_index=True)
    print(final_df.head())  # Display first few rows of the final DataFrame.
    
    final_df.to_csv('output_combined_regions_daily.csv', index=False)  # Save combined output
    
else:
    print("No data collected from any regions.")

print("Final file has been generated.")


# %%
# import pandas as pd
# final_df = pd.read_csv('output_combined_regions_daily.csv')

# %%
final_df

# %%
import pandas as pd
from datetime import datetime, timedelta

final_df['DEMAND DATE'] = pd.to_datetime(final_df['DEMAND DATE'], errors='coerce')
final_df['DEMAND TIME'] = final_df['DEMAND TIME'].astype(str)

current_time = datetime.now()
past_24_hours = current_time - timedelta(hours=24)

today_date_str = current_time.strftime("%d-%m-%y")
yesterday_date_str = (current_time - timedelta(days=1)).strftime("%d-%m-%y")

final_df['DEMAND DATETIME'] = final_df['DEMAND DATE'].astype(str) + ' ' + final_df['DEMAND TIME']
final_df['DEMAND DATETIME'] = pd.to_datetime(final_df['DEMAND DATETIME'], errors='coerce')

filtered_df = final_df[final_df['RAKE CMDT'].isin(['M', 'DOC'])]

filtered_df = filtered_df[
    (filtered_df['DEMAND DATETIME'] >= past_24_hours) & 
    (filtered_df['DEMAND DATETIME'] <= current_time)
]

# Additionally, check if the DEMAND DATE is today or yesterday
filtered_df = filtered_df[
    (filtered_df['DEMAND DATETIME'].dt.strftime("%d-%m-%y") == today_date_str) |
    (filtered_df['DEMAND DATETIME'].dt.strftime("%d-%m-%y") == yesterday_date_str)
]

filtered_df


# %% [markdown]
# # to be included in the final message

# %%
total_indents = len(filtered_df)

count_M = filtered_df[filtered_df['RAKE CMDT'] == 'M'].shape[0]

count_DOC = filtered_df[filtered_df['RAKE CMDT'] == 'DOC'].shape[0]

unique_regions = filtered_df['Region'].unique()

print(f"Total Indents Placed: {total_indents}")
print(f"Number of 'M' in RAKE CMDT: {count_M}")
print(f"Number of 'DOC' in RAKE CMDT: {count_DOC}")
print(f"Unique Regions: {unique_regions.tolist()}")


# %% [markdown]
# # Mapping 

# %%
# Load the mapping CSV files
division_mapping = pd.read_csv('division_mapping.csv')
station_names = pd.read_csv('station_names.csv')
consignee_names = pd.read_csv('consignee_names.csv')

# Create mapping dictionaries
division_dict = dict(zip(division_mapping['Short Form'], division_mapping['Full Form']))
station_dict = dict(zip(station_names['Short Form'], station_names['Full Form']))
consignee_dict = dict(zip(consignee_names['Short Form'], consignee_names['Full Form']))

# Map 'DVSN' column using division mapping
filtered_df['DVSN'] = filtered_df['DVSN'].map(division_dict).fillna(filtered_df['DVSN'])

# Map 'STTN FROM' and 'DSTN' columns using station names mapping
filtered_df['STTN FROM'] = filtered_df['STTN FROM'].map(station_dict).fillna(filtered_df['STTN FROM'])
filtered_df['DSTN'] = filtered_df['DSTN'].map(station_dict).fillna(filtered_df['DSTN'])

# Map 'CNSR' and 'CNSG' columns using consignee names mapping
filtered_df['CNSR'] = filtered_df['CNSR'].map(consignee_dict).fillna(filtered_df['CNSR'])
filtered_df['CNSG'] = filtered_df['CNSG'].map(consignee_dict).fillna(filtered_df['CNSG'])

# Display the updated DataFrame
print(filtered_df.head())

# Save the updated DataFrame if needed
filtered_df.to_csv('updated_filtered_df.csv', index=False)


# %%
filtered_df[['DVSN','STTN FROM','DSTN','CNSR','CNSG','RAKE CMDT','DEMAND DATE']]

# %% [markdown]
# # frieght rates scrapping 

# %%
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

# Setup Chrome options
options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--start-maximized")

# Initialize webdriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the URL
url = 'https://www.fois.indianrail.gov.in/FOISWebPortal/pages/FWP_FrgtCalcNew.jsp'
driver.get(url)

# Wait for the page to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[1]/input')))

# Initialize an empty list to store all row data
all_rows_data = []

# Loop through the "STTN FROM", "DSTN", "INDENTED TYPE", "RAKE CMDT" values in the CSV and process them
for index, row in df.iterrows():
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
    if rake_cmdt == "M":
        food_grains_input.clear()
        food_grains_input.send_keys("FOODGRAINS,FLOURS AND PULSES")
    elif rake_cmdt == "DOC":
        food_grains_input.clear()
        food_grains_input.send_keys("OIL CAKES AND SEEDS")
    food_grains_input.send_keys(Keys.ENTER)

    # Conditional input for "MAIZE  " or "DE-OILED CAKES"
    maize_input = driver.find_element(By.XPATH, '/html/body/div[4]/form/div/div[1]/div[1]/div[2]/div[6]/input')
    if rake_cmdt == "M":
        maize_input.clear()
        maize_input.send_keys("MAIZE  ")
    elif rake_cmdt == "DOC":
        maize_input.clear()
        maize_input.send_keys("DE-OILED CAKES")
    maize_input.send_keys(Keys.ENTER)
    time.sleep(1)
    maize_input.send_keys(Keys.ENTER)

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
    else:
        print(f"No data found for row {index + 1}")
        continue

    # Extract data from row 8 (specific column)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/table/tbody/tr[8]')))
    heading = driver.find_element(By.XPATH, '/html/body/div[1]/table/tbody/tr[8]/td[1]').text
    value = driver.find_element(By.XPATH, '/html/body/div[1]/table/tbody/tr[8]/td[3]').text
    value = value.replace(",", "")
    
    # Append data
    row_data.append(float(value))
    all_rows_data.append([station_from, station_to] + row_data)

    # Switch back to default content
    driver.switch_to.default_content()

# Create DataFrame
columns = ['Station From', 'Station To', 'Charge Name', 'Wagon Load', 'Train Load', heading]
df_result = pd.DataFrame(all_rows_data, columns=columns)

# Convert columns to numeric
df_result['Train Load'] = pd.to_numeric(df_result['Train Load'].str.replace(",", ""), errors='coerce')
df_result[heading] = pd.to_numeric(df_result[heading], errors='coerce')

# Debugging output
print("Train Load Column:")
print(df_result['Train Load'])
print(f"{heading} Column:")
print(df_result[heading])

# Calculate Per Quintal Charges
if df_result['Train Load'].isna().any() or df_result[heading].isna().any():
    print("Missing or invalid data detected.")
else:
    df_result['Per Quintal Charges'] = (df_result['Train Load'] / (df_result[heading] * 10)).round(2)
    print("Per Quintal Charges calculated successfully.")

# Save to CSV
df_result.to_csv("updated_daily_indents_freight_info.csv", index=False)

# Close driver
driver.quit()


# %%
df_result = pd.read_csv("updated_daily_indents_freight_info.csv")
df_result

# %%
import pandas as pd
import os
import pickle
from datetime import datetime
from telegram import Bot
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops in Jupyter Notebook
nest_asyncio.apply()

# Define the path for your pickle file and other configurations
pickle_file_path = 'data.pkl'
log_file_path = 'sent_rows_log.pkl'  # Log file to keep track of sent rows
BOT_TOKEN = "7836500041:AAHOL2jJ8WGrRVeAnjJ3a354W6c6jgD22RU"
CHAT_IDS = {
    8147978368: "Mohan FarmIndia",
    499903657: "Mohan Personal",
    7967517419: "Rasheed",
    7507991236: "Vidish",
    8192726425: "Rishi"}


# Step 1: Load existing data from pickle if it exists.
if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        master_data = pickle.load(file)
        master_df = pd.DataFrame(master_data)  # Convert loaded data to DataFrame
else:
    master_df = pd.DataFrame()  # Create an empty DataFrame if no file exists

# Load sent rows log if it exists
if os.path.exists(log_file_path):
    with open(log_file_path, 'rb') as file:
        sent_rows_log = pickle.load(file)
        sent_rows_df = pd.DataFrame(sent_rows_log)  # Convert loaded log to DataFrame
else:
    sent_rows_df = pd.DataFrame(columns=['S.No.'])  # Create an empty DataFrame for logs

# Step 2: Standardize 'VIA' and 'SUPPLIED TIME' columns
def standardize_columns(df):
    if 'VIA' in df.columns:
        df['VIA'] = df['VIA'].replace({pd.NA: None, '': None})
    if 'SUPPLIED TIME' in df.columns:
        df['SUPPLIED TIME'] = df['SUPPLIED TIME'].replace({pd.NA: None, '': None})
    return df

# Apply standardization to both DataFrames
master_df = standardize_columns(master_df)
filtered_df = standardize_columns(filtered_df)

# Step 3: Identify new rows that are not in master_df and not already sent
if not master_df.empty:
    new_rows = filtered_df[~filtered_df.apply(tuple, axis=1).isin(master_df.apply(tuple, axis=1))]
else:
    new_rows = filtered_df  # If master_df is empty, all rows in filtered_df are new

# Filter out rows that have already been sent based on S.No.
new_rows_to_send = new_rows[~new_rows['S.No.'].isin(sent_rows_df['S.No.'])]

async def send_alerts(new_rows):
    if not new_rows.empty:
        # Calculate summary information
        total_indents = len(filtered_df)
        count_M = filtered_df[filtered_df['RAKE CMDT'] == 'M'].shape[0]
        count_DOC = filtered_df[filtered_df['RAKE CMDT'] == 'DOC'].shape[0]
        unique_regions = filtered_df['Region'].unique()

        # Prepare the summary message
        summary_message = (
            f"*All India Indents Placed(<24hrs):* {total_indents}\n"
            f"*Total Maize RAKES:* {count_M}\n"
            f"*Total DOC RAKES:* {count_DOC}\n"
            f"*Unique Regions:* {', '.join(unique_regions.tolist())}\n\n"
        )

        # Group new rows by Demand Date
        grouped_rows = new_rows.groupby('DEMAND DATE')

        message = "*New Entries Alert:*\n\n" + summary_message
        
        for demand_date, group in grouped_rows:
            message += f"*Demand Date:* {demand_date.strftime('%Y-%m-%d')}\n"  # Format date for better readability
            
            for index, row in group.iterrows():
                # Assuming 'Per Quintal Charges' is part of the row, you can adjust the key name if necessary.
                per_quintal_charges = df_result.loc[(df_result['Station From'] == row['STTN FROM']) & 
                                                    (df_result['Station To'] == row['DSTN']), 'Per Quintal Charges'].values
                per_quintal_charges_text = f"FRT: {per_quintal_charges[0] if per_quintal_charges.size > 0 else 'N/A'}"
                serial_number = index + 1  # Serial number starts from 1, incrementing with each row
                message += (
                    f"  *Serial Number:* {serial_number}\n"  # Replace dash with serial number
                    f"    *From:* {row['STTN FROM']}\n"
                    f"    *To:* {row['DSTN']}\n"
                    f"    {per_quintal_charges_text}\n"  # Add Per Quintal Charges here
                    f"    *CMDT:* {row['RAKE CMDT']}\n"
                    f"    *CNSR:* {row.get('CNSR', 'N/A')}\n"   # Use .get() to avoid KeyError if column is missing
                    f"    *CNSG:* {row.get('CNSG', 'N/A')}\n"   # Use .get() to avoid KeyError if column is missing
                    f"    *DVSN:* {row.get('DVSN', 'N/A')}\n\n"
                )
            message += "\n"  # Add a newline after each group for spacing

        print("Prepared message:", message)  # Print for debugging
        
        bot = Bot(token=BOT_TOKEN)
        
        for chat_id, name in CHAT_IDS.items():
            try:
                print(f"Sending message to {name} (Chat ID: {chat_id})...")
                await bot.send_message(chat_id=chat_id, text=f"{name}, {message}", parse_mode='Markdown')
                print(f"Message sent to {name}.")
                
                # Log the S.No. of the sent row(s)
                for index, row in group.iterrows():
                    if row['S.No.'] not in sent_rows_df['S.No.'].values:
                        sent_rows_df.loc[len(sent_rows_df)] = {'S.No.': row['S.No.']}
                    
            except Exception as e:
                print(f"An error occurred while sending message to {name}: {e}")

async def main():
    await send_alerts(new_rows_to_send)

# Ensure this line is executed only if this script is run directly.
if __name__ == "__main__":
    asyncio.run(main())

# Step 5: Save back to pickle including the newly found rows.
combined_df = pd.concat([master_df, filtered_df], ignore_index=True)

# Drop duplicates based on all columns except 'S.No.'
combined_df = combined_df.loc[:, combined_df.columns != 'S.No.'].drop_duplicates()

# Add back the S.No. column from the original combined DataFrame for reference (if needed)
combined_df['S.No.'] = pd.Series(range(1, len(combined_df) + 1))

with open(pickle_file_path, 'wb') as file:
    pickle.dump(combined_df.to_dict(orient='records'), file)  # Save as list of dictionaries

# Save the log of sent messages back to a pickle file
with open(log_file_path, 'wb') as file:
    pickle.dump(sent_rows_df.to_dict(orient='records'), file)  # Save as list of dictionaries

print("Master pickle file has been updated with unique rows.")


# %% [markdown]
# # ----------------------------------------------------------------

# %%
import pandas as pd
import os
import pickle

# Define the path for your SMS log file
log_file_path = 'sent_rows_log.pkl'

# Check if the log file exists
if os.path.exists(log_file_path):
    # Load the log data from the pickle file
    with open(log_file_path, 'rb') as file:
        sent_rows_log = pickle.load(file)
        sent_rows_df = pd.DataFrame(sent_rows_log)  # Convert loaded log to DataFrame

    # Print the contents of the SMS logs
    print("SMS Logs:")
    print(sent_rows_df)
else:
    print("No SMS logs found. The log file does not exist.")


# %%
combined_df[['DVSN', 'STTN FROM', 'DEMAND NO.', 'DEMAND DATE', 'DEMAND TIME',
       'Expected loading date', 'CNSR', 'CNSG']]

# %%
combined_df[['CMDT', 'TT', 'PC', 'PBF',
       'VIA', 'RAKE CMDT', 'DSTN', 'INDENTED TYPE', 'INDENTED UNTS',
       'INDENTED 8W', 'OTSG UNTS', 'OTSG 8W', 'SUPPLIED UNTS', 'SUPPLIED TIME',
       'Region', 'DEMAND DATETIME']]

# %%
import pandas as pd; pd.DataFrame(pickle.load(open('data.pkl', 'rb'))) if os.path.exists('data.pkl') else print("Master file does not exist.")


# %%



