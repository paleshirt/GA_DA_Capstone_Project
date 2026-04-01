import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

shows = {
    "Hidden Brain": "20Gf4IAauFrfj7RBkjcWxh",
    "CNA Deep Dive":    "2hcojizvVOLz8dTRblRuSC"
}

all_episodes = []

for podcast_name, show_id in shows.items():
    offset = 0
    while True:
        results = sp.show_episodes(show_id, limit=50, offset=offset)
        episodes = results["items"]
        if not episodes:
            break
        for ep in episodes:
            all_episodes.append({
                "podcast":      podcast_name,
                "title":        ep["name"],
                "description":  ep["description"],
                "duration_ms":  ep["duration_ms"],
                "release_date": ep["release_date"],
                "spotify_url":  ep["external_urls"]["spotify"]
            })
        offset += 50
        print(f"{podcast_name}: fetched {offset} episodes...")

df = pd.DataFrame(all_episodes)

# Balance the classes by capping Hidden Brain at 200
hb = df[df["podcast"] == "Hidden Brain"].sample(n=200, random_state=42)
cna = df[df["podcast"] == "CNA Deep Dive"]
df_balanced = pd.concat([hb, cna]).reset_index(drop=True)

df_balanced.to_csv("data_raw/episodes_raw.csv", index=False)
print(f"\nTotal saved: {len(df_balanced)}")
print(df_balanced["podcast"].value_counts())