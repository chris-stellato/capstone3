### Proposal 1: Crowd level detection for outdoor recreation

The goal of this project would be to analyze webcam feeds from surf spots, ski slopes or public skate parks to determine the number of individuals in a given space. These three environments (water, snow, concrete) have relatively homogenous backdrops (compared to a cityscape or other more complex environments) that could lend themselves to successful image analysis. In addition to accessing crowd levels and outputting a headcount, the application could pull in other data sources like swell height, recent snowfall, weather and more to create a more complete picture of what recreationalists might encounter when they go seeking adventure. 

#### Techniques could include: 
* Image analysis
* Web scraping to pull in additional condition details
* Machine Learning to predict upcoming crowd levels
* Delivering an experience rating and alert when optimal conditions and crowd levels are present
* AWS setup to host and run model
* Flask app to interact with the system

#### Dataset
Captured and scraped from surfline.com premium, and other available public webcams from ski resorts and skate parks. 



### Proposal 2:  Black and White Photograph Colorization

The goal of this project would be to take black and white photographs of individuals and attempt to colorize the photographs, giving people a chance to see family members who have passed on in a way they haven’t been able to do in the past. This project would have a personal connection as my grandmother passed away last year, the last of my grandparents to pass on. My mother only has black-and-white images of her mother and father in their childhood and early adulthood, and I think it would be interesting to try to recreate some of these images in color to create a better visual of how it may have looked to actually be present when these black and white photographs were taken.  

This could include a flask app component where people could upload photos of their relatives for colorization, and then select the “best fit” through several iterations as the model hones in on the best hair color, eye color, and skin tone that represents their deceased relative. 

As an alternative to colorizing photos of humans, this project could pivot to colorize old black and white surf photographs, using location-specific photos to help the model accurately colorize the hue of the water, which can vary widely from location to location. 

#### Techniques could include: 
* Image color transformation
* Machine Learning to learn from existing photo sets
* AWS setup to host and run model
* Flask app to provide interactivity of uploading your own photos and selecting best fits

#### Dataset
Combining multiple datasets that are images of humans. Special attention should be paid to make sure the data set contains an extremely wide range of skin tones, hair colors, and eye colors, to ensure that the dataset could help recreate images of people of all races. 



### Proposal 3: Predicting river flows using historical flow and snowpack data

The goal of this project would be to make predictions about future river flows and water levels using historical river flow data and current/recent snowpack data and upcoming weather forecasts. The area of flow prediction already has many experts dedicated to this field of study, but predictions are done in a more manual way using historical knowledge and reading upstream gauges to predict timing. These predictions could not only be useful for outdoor recreationalists, but could also be applied to property and lives in flood-prone areas. 

#### Techniques could include: 
* Machine Learning to predict upcoming river flows (including a time-series component) 
* Web scraping to pull in additional features when api or csv downloads aren’t available
* Image analysis to read in old scanned PDFs that couldn’t be found in plaintext
* AWS setup to host and run model
* Flask app to provide interactivity, maybe something where a user could upload data from their own local river if the model is not already trained on that region? 

#### Dataset
Still seeking to understand the best combination of datasets for this project, however there is ample historical riverflow and snowpack data available for multiple regions of interest. For example the South Platte river has easily accessible historical data dating back to 1895: https://dwr.state.co.us/Tools/Stations/PLADENCO

 

