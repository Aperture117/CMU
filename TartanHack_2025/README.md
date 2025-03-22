# Crisis Response Assistance Chatbot

## Description
This chatbot is designed to provide crisis response assistance to users. The chatbot uses various APIs to offer services such as:

- Providing information on emergency procedures like CPR or what to do in case of a fire.
- Helping users find nearby emergency services like hospitals, police stations, schools, and fire stations using the Google Maps Places API.
- Displaying the latest emergency-related news via the News API.
- Utilizing OpenAI's Deepseek AI for advanced emergency query handling and routing suggestions.

The chatbot also allows users to input their location or use their IP address to get nearby emergency services and guidance during crises.

---

## Features
- **Emergency Assistance**: Provides advice and steps for common emergencies like medical emergencies or fires.
- **Find Nearby Help**: Uses the user's location to suggest nearby emergency services (hospitals, police stations, etc.).
- **Latest News**: Retrieves and filters the latest emergency news headlines from the News API.

---

## Requirements

### Libraries
- **openai** – For interacting with the OpenAI API to handle emergency-related queries.
- **requests** – To fetch data from various APIs such as Google Maps and News API.
- **pandas** – For data handling (if needed).
- **re** – For regular expression operations (standard Python library).
- **googlemaps** – For interfacing with the Google Maps API to get nearby places.
- **streamlit** – For creating the interactive web-based user interface.
- **time** – For handling time delays in the chatbot (standard Python library).
- **threading** – To handle asynchronous API requests in parallel (standard Python library).
- **urllib** – For working with URLs (standard Python library).
- **json** – For parsing JSON data (standard Python library).

---

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install openai requests pandas googlemaps streamlit

APIs and Keys:
You will need the following API keys:
OpenAI API Key: To interact with OpenAI’s GPT model for querying and generating responses.
Google Maps API Key: For location services, including fetching nearby emergency places.
News API Key: To fetch the latest news articles related to emergencies.
Ensure that the placeholders in the script (YOUR_OPENAI_API_KEY, YOUR_GOOGLE_MAPS_API_KEY, YOUR_NEWS_API_KEY) route to your respective API keys in the .env file in the same directory.
API_template.env is a template .env file to store the user's API keys

Usage:
Running the Script:
To run the Streamlit chatbot interface, use the following command:

bash
Copy
Edit
streamlit run ScottyCrew.py
This will open a web interface in your browser where you can interact with the chatbot.

Interacting with the Chatbot:
Emergency Assistance: Type your emergency query in the text area and hit "Get Help" for the chatbot to provide instructions and the nearest emergency locations.
Find Nearby Help: Select the type of emergency service (hospital, police, fire station, etc.) and click "Find Places" to view the top nearby places.
Latest News: Enter a search term to filter the latest emergency news articles or view all available news articles related to emergencies.
Running in Console:
You can also interact with the chatbot in the terminal by running the main() function, which prompts the user for input and provides a response.

Example:
When using the chatbot interface, you might get responses like this:

User: "How to perform CPR?"
Bot: "1. Call for help. 2. Begin chest compressions by placing hands in the center of the chest and pressing down hard and fast. 3. Continue CPR until help arrives."

Code Structure:
Main Logic:
The code uses Streamlit to create an interactive UI where users can input their queries.
The google_place function fetches nearby emergency services using the Google Places API.
The OpenAI API is utilized to generate smart responses based on the user's emergency query.
Emergency Assistance:
Uses a predefined set of questions and OpenAI to process emergency queries.
Find Nearby Help:
Based on the user’s current location (fetched via Google Maps Geolocation API), the chatbot provides nearby emergency services.
News API:
Fetches the latest emergency news and allows filtering by keywords.

Known Issues:
The location fetching can sometimes fail if the user's IP address is not available, and the fallback for manual location entry may not work seamlessly.
The script requires valid API keys (stored in .env file in the same directory as this Python script) and may not work without them.
API_template.env is a template .env file to store the user's API keys
Since it interacts with multiple APIs, any downtime or issue with external services may impact functionality.
