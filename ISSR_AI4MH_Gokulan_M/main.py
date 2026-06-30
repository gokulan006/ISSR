import pandas as pd
import numpy as np
import praw
import re
import os
import tensorflow as tf
from tensorflow import keras
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
import nltk
import spacy
from geopy.geocoders import Nominatim
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from dash_app import create_dashboard
from dotenv import load_dotenv
from pathlib import Path
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import time
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

client_id=os.getenv('CLIENT_ID')
client_secret=os.getenv('CLIENT_SECRET')
username=os.getenv('USER_NAME')
password=os.getenv('PASSWORD')
user_agent=os.getenv('USER_AGENT')


reddit=praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)


app= Flask(__name__)
db_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),'posts.db')
app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///' + db_path  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# Initialize Database

class RedditPosts(db.Model):
    __tablename__ = 'reddit_posts'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    post_id = db.Column(db.String(50), unique=True, nullable=False)
    username = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime , nullable=False)
    content = db.Column(db.Text)
    upvotes = db.Column(db.Integer)
    comments = db.Column(db.Integer)
    sentiment = db.Column(db.String(50))
    risk_level = db.Column(db.String(50))

class PostLocations(db.Model):
    __tablename__= 'locations'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float(), nullable=False)
    longitude = db.Column(db.Float(), nullable=False)

class UserBehavior(db.Model):
    __tablename__ = 'Users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    post_count = db.Column(db.Integer, default=0)
    high_risk_count = db.Column(db.Integer, default=0)
    sentiment_trend = db.Column(db.Float(), default=0.0)
    timestamp = db.Column(db.DateTime, nullable=False)

with app.app_context():
    db.create_all()

dash_app = create_dashboard(app)

# Setting up Geolocation Extractor
nlp=spacy.load('en_core_web_sm')
nltk.download('stopwords')
STOPWORDS= set(stopwords.words('english'))
geolocator= Nominatim(user_agent='crisis_detector')

# Setting up Risk Classifier Model
risk_labels= ['High Risk', 'Low Risk', 'Moderate Risk']   


# Loading the model and tokenizer
def load_risk_classification_model():
    base_dir=Path('risk_model_package')

    tokenizer_dir=base_dir / 'tokenizer'
    
    tokenizer = DistilBertTokenizerFast(
        vocab_file=str(tokenizer_dir/'vocab.txt'),
        tokenizer_config_file=str(tokenizer_dir/'tokenizer_config.json')
    )

    model=tf.keras.models.load_model(
        str(base_dir/'model'),
        custom_objects={
            "TFDistilBertModel": TFDistilBertModel,
            "Attention": tf.keras.layers.Attention
        }
    )

    return tokenizer, model

tokenizer, model=load_risk_classification_model()

# Sentiment Analysing Model
analyser = SentimentIntensityAnalyzer()

# Subreddits 
subreddits = [
    # Mental Health
    "mentalhealth",          
    "depression_help",

    # Suicidal
    "offmychest",
    "suicidewatch",

    # Substance Use    
    "Drugs",         
    "addiction" 
]

# Custom Lexicon keywords for filtering the posts efficiently
crisis_lexicon = {
    "mental_health": {
        "explicit": [
            "depression", "anxiety", "hopeless", "overwhelmed", "panic attack", "social anxiety", "PTSD",
            "flashbacks", "nightmares", "dissociation", "intrusive thoughts", "trauma", "worthless", "numb",
            "empty", "self-hate", "isolated", "exhausted", "insomnia", "brain fog", "mood swings", "crying",
            "irritable", "emotional pain", "stress", "burnout", "mental breakdown", "disconnected", "lonely",
            "fear", "can’t breathe", "racing thoughts", "uncontrollable thoughts", "guilt", "shame",
            "unmotivated", "feeling lost", "no energy", "feeling stuck", "lack of focus", "can’t function",
            "losing interest", "fatigue", "restless", "self-doubt", "low self-esteem", "can’t concentrate",
            "overthinking", "avoiding people", "heart racing", "bipolar disorder", "manic episode", "depressive episode",
            "psychosis", "schizophrenia", "dissociative identity disorder", "borderline personality disorder", "OCD",
            "eating disorder", "self-harm", "cutting", "burning", "scratching", "hair pulling", "skin picking"
        ],
        "coded": [
            "low spoons", "heavy chest", "stuck in my head", "faking smiles", "pretending to be okay",
            "can’t mask anymore", "floating away", "trapped in my mind", "checked out", "shutting down",
            "mind is dark", "crawling skin", "no escape", "lost in a loop", "echo chamber", "NPC",
            "ghost mode", "just background noise", "permanent headache", "seeing static", "grey days",
            "can’t wake up", "everything is too much", "drowning in thoughts", "the void", "static noise in my head",
            "too tired to exist", "mind racing nonstop", "watching from the outside", "just surviving",
            "in my bubble", "white noise brain", "time slipping away", "everything feels fake", "no feeling left",
            "pretending to care", "nobody notices", "losing myself", "shadow of myself"
        ]
    },
    "suicide": {
        "explicit": [
            "suicide", "end my life", "kill myself", "no reason to live", "want to die", "take my life",
            "self-harm", "overdose", "jump off", "hang myself", "poison myself", "gun to my head",
            "rope", "final decision", "permanent solution", "giving up", "goodbye message", "can’t keep going",
            "done fighting", "no more pain", "too much to handle", "empty inside", "tired of everything",
            "can’t escape", "no future", "last chance", "no point anymore", "nobody would care", "the pain won",
            "losing control", "final thoughts", "goodbye world", "writing a note", "checking out",
            "sleep forever", "want to disappear", "can’t go on", "ready to leave", "the end is near",
            "don’t wake up", "fading away", "unbearable pain", "crushed inside", "nothing left to give",
            "nothing left to lose", "life is pointless", "done with everything", "no second chances",
            "only one way out"
        ],
        "coded": [
            "KMS", "unalive", "permanent nap", "going to sleep forever", "catching the bus", "rope game",
            "taking the exit", "no respawn", "checking out early", "logging off for good", "closing my book",
            "gone fishing", "see you on the other side", "one way trip", "empty chair soon", "no more reruns",
            "lights out", "fading signal", "CTRL + ALT + DELETE", "end scene", "GG", "long ride", "no tomorrow",
            "last sunrise", "countdown started", "signing off", "done pretending", "no more next time",
            "too tired to restart", "can’t respawn this time", "silent mode forever", "taking my final bow",
            "no more chapters left", "final logout", "going ghost", "shutting down for good",
            "see you in another life", "just a memory soon", "yeet", "CTB", "SH", "final yeet", "taking the L",
            "logging off", "permanent sleep", "no respawn", "final exit", "see you on the other side", "one way trip",
            "lights out", "fading signal", "CTRL + ALT + DELETE", "end scene", "GG", "long ride", "no tomorrow",
            "last sunrise", "countdown started", "signing off", "done pretending", "no more next time",
            "too tired to restart", "can’t respawn this time", "silent mode forever", "taking my final bow",
            "no more chapters left", "final logout", "going ghost", "shutting down for good",
            "see you in another life", "just a memory soon"
        ]
    },
    "substance_use": {
        "explicit": [
            "alcoholic", "drunk", "high", "cocaine", "heroin", "meth", "pills", "painkillers", "Xanax",
            "fentanyl", "addiction", "rehab", "withdrawal", "overdose", "OD", "drugged out", "binge drinking",
            "blackout", "opiates", "stimulants", "narcotics", "prescription abuse", "substance abuse",
            "dealer", "relapse", "detox", "cold turkey", "mixing drugs", "taking too much", "need another hit",
            "pushing limits", "can’t quit", "always craving", "hooked", "strung out", "need a fix",
            "chasing the high", "spiraling", "downward spiral", "numb the pain", "out of control",
            "can’t stop", "slurring speech", "losing grip", "body shutting down", "no more control",
            "shaky hands", "nightly drinking", "losing memory", "fentanyl", "methamphetamine", "crack cocaine",
            "LSD", "ecstasy", "ketamine", "mushrooms", "spice", "synthetic opioids", "bath salts", "GHB",
            "roofies", "date rape drugs", "inhalants", "nitrous oxide"
        ],
        "coded": [
            "snow", "smack", "molly", "lean", "benzos", "bars", "percs", "dope", "zaza", "nod", "candy flipping",
            "skittles", "chasing dragons", "plug", "gassed", "geeked", "wired", "faded", "blow", "black tar",
            "speedball", "cloud 9", "laced", "ripped", "popped a bean", "shot up", "420", "bump",
            "plug hit me up", "one more round", "sippin’", "cooking up", "jib", "Xannies", "tweak",
            "on a bender", "too deep", "red eyes"
        ]
    }
}

keywords = (
    crisis_lexicon['mental_health']['explicit'] + 
    crisis_lexicon['mental_health']['coded'] + 
    crisis_lexicon['substance_use']['explicit'] + 
    crisis_lexicon['substance_use']['coded'] +
    crisis_lexicon['suicide']['explicit'] + 
    crisis_lexicon['suicide']['coded']
)

 
def extract_reddit_posts(limit=2000):
    posts_data = []
    try: 
        for subreddit_name in subreddits:
            print(f"Fetching posts from r/{subreddit_name}...")
            subreddit = reddit.subreddit(subreddit_name)

            for post in subreddit.new(limit=limit):
                timestamp=post.created_utc
                timestamp=pd.to_datetime(timestamp,unit='s')
                if any(keyword.lower() in post.title.lower() + post.selftext.lower() for keyword in keywords):
                    posts_data.append([
                        post.id,
                        str(post.author),
                        post.created_utc,
                        post.title,
                        post.selftext,
                        post.score,
                        post.num_comments
                    ])
        
        columns=['post_id', 'username','timestamp','title','content','upvotes','comments']
        df=pd.DataFrame(posts_data, columns=columns)
        
        return df
    
    except Exception as e:
        print("An error occured: ",e)

def update_timestamp(timestamp):
    return pd.to_datetime(timestamp,unit='s')

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'http\S+', '', text)                                      # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)                                  # Remove special characters
    text = re.sub(r'\d+', '', text)                                          # Remove numbers
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords

    return text

def sentiment_scores(sentence):

    if isinstance(sentence, float):
        sentence = str(sentence)
    
    sentence_dict = analyser.polarity_scores(sentence)

    if sentence_dict['compound'] >=0.10:
        return 'Positive'
    elif sentence_dict['compound'] <=-0.10:
        return 'Negative'
    else:
        return 'Neutral'
    

def predict_risk_level(text):
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    
    probs = model.predict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    
    predicted_class = np.argmax(probs, axis=-1)[0]

    return risk_labels[predicted_class]

# Function to extract location

def extract_location(text):
    if isinstance(text,str):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'GPE' or ent.label_ == 'LOC':
                return ent.text

    return None        
    
def geocode_location(place):
    try:
        
        location = geolocator.geocode(place, addressdetails=True,language='en',timeout=10)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            address = location.raw.get('address',{})
            city=address.get('city') or address.get('town') or address.get('village')
            state=address.get('state')
            country=address.get('country')
            location_mapped= city or state or country

            return pd.Series({'latitude':latitude,'longitude':longitude,'location':location_mapped})
        
        return pd.Series({'latitude':None,'longitude':None,'location':None})
    except Exception as e:
        print("Error occured while extracting coordinates :",e)
        return pd.Series({'latitude':None,'longitude':None,'location':None})
    
def store_df_in_sql(df):
    with app.app_context():
        for _,row in df.iterrows():
            post_exists = RedditPosts.query.filter_by(post_id=row['post_id']).first()

            if not post_exists:
                post = RedditPosts(
                    post_id=row['post_id'],
                    username=str(row['username']),
                    timestamp=row['timestamp'],
                    content=row['content'],
                    upvotes=row['upvotes'],
                    comments=row['comments'],
                    sentiment=row['sentiment'],
                    risk_level=row['risk_level']
                )
                db.session.add(post)
        
        db.session.commit()


def update_user_behavior(post_id, username, content, risk_level, timestamp):
    with app.app_context():
         
        try:
            user = UserBehavior.query.filter_by(username=str(username)).first()
            post = RedditPosts.query.filter_by(post_id=post_id).first()

            if user and not post:
                user.post_count +=1
                if risk_level == 'High Risk':
                    user.high_risk_count +=1
                content=str(content)
                sentiment_score=analyser.polarity_scores(content)['compound']
                user.sentiment_trend = (user.sentiment_trend + sentiment_score)/2
                user.timestamp=timestamp
                
            elif not post:
                content=str(content)
                sentiment_score = analyser.polarity_scores(content)['compound']
                user = UserBehavior(
                    username=str(username),
                    post_count=1,
                    high_risk_count=1 if risk_level == 'High Risk' else 0,
                    sentiment_trend=sentiment_score,
                    timestamp=timestamp
                )
                db.session.add(user)
            
            db.session.commit()
            db.session.close()
            return 

        except Exception as e:
            print("Error occured while updating the user behavior: ",e)
            return 
        
def get_coordinates(df_coor):

    df_coor['location'] = df_coor['content'].progress_apply(lambda x: extract_location(str(x)))
    df_coor.dropna(inplace=True)
    df_coor['location'] = df_coor['location'].str.title()
    print("Dataframe of extracted Locations",df_coor.info())
    df_coor[['latitude','longitude','location']] = df_coor['location'].progress_apply(lambda x:geocode_location(x))
     
    df_coor.dropna(inplace=True)

    if df_coor.empty:
        print("No Valid Coordinates found.")
        return

    print("Coordinates Completed.")   
    print("Dataframe after extracting Coordinates",df_coor.info())
     
    return df_coor

def store_high_risk_locations(df_coor):
    with app.app_context():
        for _, row in df_coor.iterrows():
            existing_entry = PostLocations.query.filter_by(username=row['username']).first()

            if not existing_entry:
                entry = PostLocations(
                    username=row['username'],
                    risk_level=row['risk_level'],
                    timestamp=row['timestamp'],
                    location=row['location'],
                    latitude=row['latitude'],
                    longitude=row['longitude']
                )
                db.session.add(entry)

        db.session.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze/', methods=['GET'])
def analyze_posts():
    global df 
    df = extract_reddit_posts()
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df['content'] = df['content'].apply(preprocess_text)
    df['sentiment'] = df['content'].apply(sentiment_scores)
    df['risk_level'] = df['content'].progress_apply(predict_risk_level)
    df.dropna(inplace=True)
    print("Count of risk levels",df['risk_level'].value_counts())
    
    print("Lets start analyze the users")
    for i in tqdm(range(len(df))):
        time.sleep(0.2)
        if(df.iloc[i]['post_id']!=None):
            update_user_behavior(
                post_id=df.iloc[i]['post_id'],
                username=df.iloc[i]['username'],
                content=df.iloc[i]['content'],
                risk_level=df.iloc[i]['risk_level'],
                timestamp=df.iloc[i]['timestamp']
            )
    print("Lets store the extracted posts in Database")
    store_df_in_sql(df)
    print("Lets get the coordinates of extracted location")
    df_coor=df.copy()
    df_coor=get_coordinates(df_coor)
    print("Lets store the coordinates of the location")
    store_high_risk_locations(df_coor)

    return jsonify({"message": "Data stored in SQL successfully!", "rows": len(df)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT',5000)))


    