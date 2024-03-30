import webtech
from Wappalyzer import Wappalyzer, WebPage
import whois
import warnings
import requests
import re
warnings.filterwarnings("ignore")


# function to find web stack technology
def find_tech_stack(url):
    wt = webtech.WebTech(options={'json': True})
    try:
        # Finding tech with webtech library
        report = wt.start_from_url(url)
        techs_1 = {tech['name'] for tech in report['tech']}

        # Finding tech with wappalyzer library
        webpage = WebPage.new_from_url(url)
        wappalyzer = Wappalyzer.latest()
        techs_2 = wappalyzer.analyze(webpage)

        # Combining both the results
        return list(techs_1.union(techs_2))

    except Exception:
        return None


# Function to find timeline of updates


# Function to find information about operator
def find_operator_info(url):
    try:
        domain = whois.whois(url)
        if domain:
                        # Operator, Managing director, Telephone no., Email, Servers 
            return [domain.registrar, domain.name, domain.phone, domain.emails, domain.name_servers]
        else:
            return None
        
    except whois.parser.PywhoisError as e:
        print("Error: ", e)
        return None

    except Exception as e:
        return None


# Company analysis
def find_company_analysis(url):
    # Extract company name from URL
    pattern = r"https?://(?:www\.)?([a-zA-Z0-9-]+)\."
    match = re.search(pattern, url)
    if match:
        company_name = match.group(1)
    else:
        return None
    
    # Get company SYMBOL
    api_key = 'DQQIQ7AI5OXYSQ43'  # Replace with your Alpha Vantage API key
    alpha_url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={api_key}'
    response = requests.get(alpha_url)
    data = response.json()
    if 'bestMatches' in data and len(data['bestMatches']) > 0:
        symbol = data['bestMatches'][0]['1. symbol']
    else:
        return None
    
    # Get Company Overview
    alpha_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(alpha_url)
    data = response.json()
    if data == {}:
        return None
    else:
        return [data['Name'], data['Description'], data['Country'], data['Industry'], data['Address'], data['MarketCapitalization'], data['DividendPerShare'], data['ProfitMargin'], data['RevenueTTM'], data['GrossProfitTTM']]


# find_company_analysis("https://cnn.com")