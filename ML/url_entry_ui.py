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
from ml_inference import predict_from_user


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


def result_url():
    st.subheader("URL Based Input")
    st.write("Paste a job post URL from Indeed, LinkedIn, or Naukri")

    job_url = st.text_input("Paste Job URL")

    result = None
    if st.button("CHECK"):
        if job_url.strip():
            with st.spinner("Scraping job details..."):
                try:
                    title, company, description = job_scraper(job_url)

                    st.subheader("Job Title")
                    st.write(title)

                    st.subheader("Company Name")
                    st.write(company)


                    st.divider()
                    st.subheader("AI Authenticity Analysis")

                    if description != "Not found":
                        with st.spinner("Analyzing job description..."):
                            # Call the actual ML model
                            result = predict_from_user(description)
                        
                        if result["status"] == "FAKE":
                            st.error(f"🚨 Warning: This job post looks FAKE.")
                            st.info("Reasoning: The description contains patterns often associated with fraudulent job advertisements.")
                            
                            if result["words"]:
                                st.warning(f"Suspicious words found: {', '.join(result['words'])}")
                        else:
                            st.success(f"✅ Verified: This job post looks REAL.")
                    else:
                        st.warning("Insufficient data to perform authenticity analysis.")
                except Exception as e:
                    st.error(f"An error occurred during scraping: {e}")
        else:
            st.warning("Please paste a valid job URL")

    return result

if __name__ == "__main__":
    st.set_page_config(page_title="URL Entry ", layout="centered")
    result_url()
