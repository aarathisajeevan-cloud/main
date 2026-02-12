# ---------- IMPORTS ----------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import streamlit as st
import requests


# ---------- STEP 1: WEBSITE DETECTOR ----------
def detect_site(url):
    domain = urlparse(url).netloc.lower()

    if "indeed" in domain:
        return "indeed"
    elif "linkedin" in domain:
        return "linkedin"
    elif "naukri" in domain:
        return "naukri"
    else:
        return "generic"


# ---------- STEP 2: SCRAPER ----------
def job_scraper(url):
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.minimize_window()
    driver.set_window_position(-10000,0)


    driver.get(url)
    time.sleep(10)   # wait for page load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    site = detect_site(url)

    title = "Not found"
    company = "Not found"
    description = "Not found"

    # ---------- INDEED ----------
    if site == "indeed":
        # Job Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Not found"

        # Company Name (UPDATED)
        company_tag = soup.find(
            "div", attrs={"data-testid": "inlineHeader-companyName"}
        )
        company = company_tag.get_text(strip=True) if company_tag else "Not found"

        # Job Description
        desc_tag = soup.find("div", id="jobDescriptionText")
        description = (
            desc_tag.get_text(" ", strip=True) if desc_tag else "Not found"
        )
        
    # ---------- LINKEDIN ----------
    elif site == "linkedin":
        # ---------- JOB TITLE ----------
        title_tag = soup.select_one("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Not found"

        # ---------- COMPANY NAME ----------
        company_tag = soup.select_one(
            "a.topcard__org-name-link, span.topcard__flavor"
        )
        company = company_tag.get_text(strip=True) if company_tag else "Not found"

        # ---------- JOB DESCRIPTION ----------
        description = "Not found"

        possible_desc_blocks = [
            "div.jobs-description__content",
            "div.jobs-box__html-content",
            "section.description",
            "div.jobs-description-content__text"
        ]

        for selector in possible_desc_blocks:
            desc_tag = soup.select_one(selector)
            if desc_tag and len(desc_tag.get_text(strip=True)) > 50:
                description = desc_tag.get_text(" ", strip=True)
                break
        # #Job title
        # title_tag = soup.select_one('div[class*="top-card"] strong')
        # title = title_tag.get_text(strip = True) if title_tag else "Not found"

        # #Company name
        # company_tag = soup.select_one('a[aria-label$="logo"]')
        # if company_tag:
        #     company = company_tag.get("aria-label").replace(" logo", "").strip()
        # else:
        #     company = "Not found"

        # #Job description
        # desc_tag = soup.select_one("div#job-details")
        # if desc_tag:
        #     description = desc_tag.get_text(" ", strip = True)
        # else:
        #     description = "Not found"


        # company_tag = soup.select_one(
        #     "a.topcard__org-name-link, span.topcard__flavor")



    # ---------- NAUKRI ----------
    elif site == "naukri":
           # Job Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Not found"

        # Company Name (UPDATED)
        company_tag = soup.select_one('div[class*="jd-header-comp-name"] a')
        if company_tag:
            company = company_tag.get_text(strip = True)

        # Job Description
        desc_tag = soup.select_one('div[class*="dang-inner-html"]')
        if desc_tag:
            description = desc_tag.get_text(" ", strip=True)
        else:
            description = "not found"
    return title, company, description


# ---------- STEP 3: STREAMLIT UI ----------
st.set_page_config(page_title="URL Entry ", layout="centered")

st.title("URL Based Input")
st.write("Paste a job post URL from Indeed, LinkedIn, or Naukri")

job_url = st.text_input("Paste Job URL")

if st.button("Scrape Job"):
    if job_url.strip():
        with st.spinner("Scraping job details..."):
            title, company, description = job_scraper(job_url)

        st.subheader("Job Title")
        st.write(title)

        st.subheader("Company Name")
        st.write(company)

        st.subheader("Job Description")
        st.write(description)
    else:
        st.warning("Please paste a valid job URL")
