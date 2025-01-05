import gym
from recsim import document
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym

import pandas as pd
import numpy as np
from datetime import datetime
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2

tourist_spots_data = pd.read_csv("data.csv")
tourist_spots_data

#tourist_spots_data['Opening Time'] = pd.to_datetime(tourist_spots_data['Opening Time']).dt.time
#tourist_spots_data['Closing Time'] = pd.to_datetime(tourist_spots_data['Closing Time']).dt.time
#tourist_spots_data

tourist_spots_data.info()

tourist_spots_data['Budget (INR)'] = tourist_spots_data['Budget (INR)'].astype(int)
tourist_spots_data.info()

feedback_data = pd.read_csv('feedback.csv')
feedback_data

feedback_data.info()

tourist_spots_data['City'].unique()

def get_location_coordinates(location_name):
    loc = Nominatim(user_agent="GetLoc")
    get_loc = loc.geocode(location_name)
    return get_loc.latitude, get_loc.longitude

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    R = 6371.0

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def store_feedback(user_id, place, feedback, rating, date):
    feedback_data = pd.DataFrame({
        'user_id': [user_id],
        'place': [place],
        'feedback': [feedback],
        'rating': [rating],
        'date_of_feedback': [date]
    })
    feedback_data.to_csv("feedback.csv", mode='a', header=False, index=False)

datetime.now().time()

class TravelEnv(gym.Env):
    def __init__(self, tourist_spots_data):
        super(TravelEnv, self).__init__()
        self.tourist_spots_data = tourist_spots_data
        self.current_state = None
        self.done = False

    def step(self, action):
        recommended_spot = self.tourist_spots_data.iloc[action]
        spot_name = recommended_spot['Tourist Spot']
        spot_opening_time = recommended_spot['Opening Time']
        spot_closing_time = recommended_spot['Closing Time']

        current_time = datetime.now().time()
        opening_time = datetime.strptime(spot_opening_time, "%H:%M:%S").time()
        closing_time = datetime.strptime(spot_closing_time, "%H:%M:%S").time()

        if opening_time <= current_time <= closing_time:
            self.current_state = spot_name
            reward = 1
        else:
            reward = -0.1  # Some penalty value

        # Check if episode is done (e.g., user reaches the budget limit)
        if self.user_budget <= 0:
            self.done = True

        return self.current_state, reward, self.done, {}

    def reset(self):
        self.current_state = None
        self.done = False
        return self.current_state

def get_recommendation(user_id, place, budget, tourist_spots_data, user_feedback):

    # Check if the user has any previous feedback
    user_history = user_feedback[user_feedback['user_id'] == user_id]
    user_has_history = not user_history.empty

    # Filter tourist spots based on location
    place_filtered = tourist_spots_data[tourist_spots_data['City'] == place]

    # Filter spots based on budget
    affordable_spots = place_filtered[place_filtered['Budget (INR)'].astype(int) <= budget]

    # Filter spots based on opening hours
    current_time = datetime.now().time()
    open_spots = affordable_spots[affordable_spots.apply(lambda x:
                                                        datetime.strptime(x['Opening Time'], "%H:%M:%S").time() <= current_time <= datetime.strptime(x['Closing Time'], "%H:%M:%S").time(), axis=1)]


    if user_has_history:
        # Merge user's history with feedback from other users
        all_feedback = pd.merge(user_feedback, user_history, how='outer')
        feedback_scores = all_feedback.groupby('tourist_spot')['rating'].mean()
        recommended_spots = open_spots.merge(feedback_scores, left_on='Tourist Spot', right_index=True, how='left')
        recommended_spots = recommended_spots.sort_values(by='rating', ascending=False)
    else:
        # If user is new, recommend based on overall ratings
        recommended_spots = open_spots.sort_values(by='rating', ascending=False)


    if not recommended_spots.empty:
        top_recommendation = recommended_spots.iloc[0]['Tourist Spot']
        return top_recommendation
    else:
        return "No recommendations available within your budget and location."

def main():
    env = TravelEnv(tourist_spots_data)
    choice_features = {
        "Tourist Spot": 0,
        "City": 1,
        "Location Lat": 2,
        "Location Long": 3,
        "Budget (INR)": 4,
        "Opening Time": 5,
        "Closing Time": 6
    }
    choice_model = MultinomialLogitChoiceModel(choice_features)
    doc = document.AbstractDocument
    recsim_env = recsim_gym.RecSimGymEnv(env, choice_model, doc)

    # Get user input
    user_id = int(input("Enter user ID: "))
    place = input("Enter desired place: ")
    budget = float(input("Enter budget (INR): "))

    # Get recommendation
    recommendation = get_recommendation(user_id, place, budget, tourist_spots_data, feedback_data)

    # Display recommendation to the user
    print("Your Recommendation:", recommendation)

    # Ask for feedback
    feedback = input("Provide feedback (Yes/No): ")
    if feedback.lower() == "yes":
        rating = int(input("Rate the recommendation (1-5): "))
        comment = input("Comments on this place: ")
        date_of_feedback = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        store_feedback(user_id, recommendation, comment, rating, date_of_feedback)

if __name__ == "__main__":
    main()





#if (place_filtered['Location Lat'] - lat) ** 2 + (place_filtered['Location Long'] - lon) ** 2 <= distance_threshold ** 2:
     #   filtered_spots = tourist_spots_data[(place_filtered['Location Lat'] - lat) ** 2 + (place_filtered['Location Long'] - lon) ** 2 <= distance_threshold ** 2]
      #  print(filtered_spots)



