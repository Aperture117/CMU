import os
from openai import OpenAI
import requests
import pandas as pd
import re
import urllib.request
import json
import googlemaps
import streamlit as st
import time
import threading
from dotenv import load_dotenv

load_dotenv()
# API keys
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# OPEN AI model from NVIDIA NIMs homepage
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= OpenAI_API_KEY
)

####################################
# Get my current location (lat, lng)
# Google Maps API

def get_current_location():
    # Try to fetch the location based on IP (as before)
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}'
    data = {'considerIp': True}  # Uses the current IP for location

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        location_data = response.json()

        if 'location' in location_data:
            return location_data['location']  # {'lat': 40.448819, 'lng': -79.953920}
        else:
            print("Error: 'location' not found in response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching location: {e}")
        return None

# Fetch the current location
current_location = get_current_location()

# If location was fetched successfully, use it
if current_location:
    print(f"Your location: {current_location}")
else:
    # Prompt user for manual input if automatic location fetching fails
    print("\nUnable to determine your current location automatically.")
    address = input("Enter your address: ")

    # Use Google Maps Geocoding API to convert the address into latitude and longitude
    geocode_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}'

    try:
        geocode_response = requests.get(geocode_url)
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()

        if geocode_data['status'] == 'OK':
            # Get the first result's latitude and longitude
            lat = geocode_data['results'][0]['geometry']['location']['lat']
            lng = geocode_data['results'][0]['geometry']['location']['lng']
            current_location = {'lat': lat, 'lng': lng}
            print(f"Your location from address '{address}': {current_location}")
        else:
            print(f"Error: Unable to geocode the address '{address}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching address geocode: {e}")


def google_place(place_type):
    # Ensure only valid types are allowed (case-insensitive)
    valid_types = ["hospital", "police", "school", "fire_station"]

    # Convert the place_type to lowercase for case-insensitive comparison
    place_type = place_type.lower()

    if place_type not in valid_types:
        print(f"Error: Invalid place type '{place_type}'. Valid types are: {', '.join(valid_types)}.")
        return None

    RADIUS = 10000  # meters
    TYPE = place_type  # hospital, police station, emergency shelter, fire station
    URL = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={current_location["lat"]},{current_location["lng"]}&radius={RADIUS}&type={TYPE}&key={GOOGLE_API_KEY}'

    response = requests.get(URL)
    if response.status_code == 200:
        results = json.loads(response.content)["results"]
        return results[:3]  # Returning top 3 places
    else:
        print(f"Error: Unable to fetch data for {place_type}.")
        return None


#### News API

def news_api():
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()

        # Check if the API response contains articles
        if data.get("status") == "ok" and "articles" in data:
            articles = []
            for article in data["articles"]:
                # Ensure each article has a title and description
                if article.get("title") and article.get("description"):
                    articles.append({
                        'title': article["title"],
                        'description': article["description"]
                    })
            return articles
        else:
            print("Error: Invalid API response format.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
    
    
### Deepseek API

def summarize_openAI(query):
    try:
        messages = [
            {"role": "system", "content": f"Using the user's {current_location} which is in current, suggest the nearest place that user ask. Provide emergency contact numbers if available."
                    "A user is in 'current_location' during a natural disaster or emergency situation. Recommend the closest emergency shelter and safe routes to reach it."
                    },

            {"role": "user", "content": f"{query}"
                                        "Do not include any explanations, reasoning, or 'thinking' sections. "
                                        "Focus only on providing the following: "
                                        "1. Steps to address the emergency. "
                                        "2. The nearest location (e.g., hospital, shelter, police station) based on the user's current location. "
                                        "3. Route guidance on how to reach the location safely from my location. "
                                        "4. Emergency contact numbers if available. "
                                        "Do not include any explanations, reasoning, or internal monologues (e.g., '<think>...</think>')."
                                        "Focus only on delivering the necessary information in a clear and direct manner."}]

        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1", # Using Deekseek R1 model
            messages=messages,
            top_p=0.5, # creativity
            max_tokens=1024, # max output tokens
            tools=[
                {
                "type": "function",
                "function" : {
                    "name": "google_place",
                    "description": "This function calls the Google Places API to find the top places of a specified type near a specific location. It can be used when a user expresses a need (e.g.,  hospital or police station) or wants to find a certain type of place.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "place_type": {
                                "type": "string",
                                "description": "The type of place to search for."
                            }
                        }
                    }

                }
            }
            ]
        )
        answer = response.choices[0].message.content
        return answer.strip()

    except Exception as e:
        print("Error calling OpenAI API:", e)
        return ""
    

####################
# UI Title
####################
  
##### Build Html output
sample_questions = [
    "How to perform CPR?",
    "What to do in case of a fire?",
    "Where is the nearest hospital?",
    "How to handle a medical emergency?",
    "Emergency contact numbers in my area",
]

st.title("🚨 Crisis Response Assistance")


# Sidebar Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.selectbox("Select an option", ["Emergency Assistance", "Find Nearby Help", "Latest News"])
    
# Sidebar with Sample Questions
st.sidebar.subheader("📝 Sample Questions")
selected_sample = st.sidebar.radio("Click a question to use:", sample_questions)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Main Area Based on Selected Menu Option
if menu == "Emergency Assistance":
    st.subheader("💬 Get Emergency Help")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
            
    # If a sample question is selected, pre-fill it in the te
    # Chat
    user_query = st.chat_input("Type your message here...")

    if st.session_state.chat_history and not user_query and selected_sample:
        user_query = selected_sample

    if user_query:
    # Append user's message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
    # Display the user message immediately
        st.chat_message("user").write(user_query)
    
    # Display elapsed time during API processing
        timer_placeholder = st.empty()
        start_time = time.time()
        timer_placeholder.markdown(f"**Time spent: 0.0 seconds**")
        
        result = {}
        def run_api():
            result['response'] = summarize_openAI(query=user_query)
        
        api_thread = threading.Thread(target=run_api)
        api_thread.start()
        
        while api_thread.is_alive():
            elapsed_time = time.time() - start_time
            timer_placeholder.markdown(f"**Time spent: {elapsed_time:.1f} seconds**")
            time.sleep(0.1)
        api_thread.join()
        timer_placeholder.empty()
        
        # Append and display the assistant's reply
        st.session_state.chat_history.append({"role": "assistant", "content": result['response']})
        st.chat_message("assistant").write(result['response'])
                

elif menu == "Find Nearby Help":
    st.subheader("📍 Find Emergency Locations")
    place_type = st.selectbox("Select emergency service", ["hospital", "police", "school", "fire_station"])
    if st.button("Find Places"):
        results = google_place(place_type)
        if results:
            st.write(f"### Top {place_type.capitalize()}s Nearby:")
            for place in results:
                st.write(f"**{place['name']}**\n📍 {place['vicinity']}")
        else:
            st.warning("No results found.")

elif menu == "Latest News":
    st.subheader("📰 Latest Emergency News")
    search_term = st.text_input("Search for a word or phrase in the news:")
    articles = news_api()
    
    if articles:
            # Filter articles based on the search term
            if search_term:
                filtered_articles = [
                    article for article in articles
                    if search_term.lower() in article['title'].lower() or search_term.lower() in article['description'].lower()
                ]
            else:
                filtered_articles = articles
                
            if filtered_articles:
                for article in filtered_articles:
                    st.write(f"**{article['title']}**\n{article['description']}")
            else:
                st.warning(f"No articles found containing the term: '{search_term}'.")
    else:
        st.warning("No news available.")

def main():

    # Build question
    question = input("How can I help you? :  \n")
    answer = summarize_openAI(question)

    print("=== User Query ===")
    print(question)
    print("=== Client Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
