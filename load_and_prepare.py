"""
load_and_prepare.py — Feature engineering for local use.
(In Colab, this is embedded in the notebook — Step 3)
Dataset: airt-ml/twitter-human-bots | CC-BY-SA 3.0
https://huggingface.co/datasets/airt-ml/twitter-human-bots
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)

RAW = "data/twitter_human_bots.parquet"
if not os.path.exists(RAW):
    print("❌ Place data/twitter_human_bots.parquet in the data/ folder first.")
    print("   Download: https://huggingface.co/datasets/airt-ml/twitter-human-bots/tree/refs%2Fconvert%2Fparquet/default/train")
    raise SystemExit(1)

print("Loading raw dataset...")
df = pd.read_parquet(RAW)
df['label'] = (df['account_type'] == 'bot').astype(int)

# ── GROUP 1: TIMING REGULARITY ────────────────────────────
df['tweets_per_day']         = df['average_tweets_per_day'].fillna(0).clip(0,1000)
df['account_age_days']       = df['account_age_days'].fillna(0).clip(0,6000)
df['total_tweets']           = df['statuses_count'].fillna(0).clip(0,3_000_000)
df['total_likes_given']      = df['favourites_count'].fillna(0).clip(0,1_000_000)
df['tweet_density']          = (df['total_tweets']/(df['account_age_days']+1)).clip(0,500)
df['likes_per_day']          = (df['total_likes_given']/(df['account_age_days']+1)).clip(0,1000)
df['timing_regularity_score']= (df['tweets_per_day']/(np.log1p(df['account_age_days'])+1)).clip(0,500)
df['lifetime_tweet_rate']    = (df['total_tweets']/(df['account_age_days']*24+1)).clip(0,100)

# ── GROUP 2: ENGAGEMENT BURST ─────────────────────────────
df['is_high_tweeter']     = (df['tweets_per_day']>50).astype(int)
df['abnormal_tweet_rate'] = (df['tweets_per_day']>100).astype(int)
df['burst_index']         = (df['total_tweets']/(df['account_age_days']+1)).clip(0,1000)
df['burst_flag']          = (df['burst_index']>20).astype(int)
df['sudden_fame']         = ((df['account_age_days']<30)&(df['followers_count']>1000)).astype(int)
df['engagement_paradox']  = ((df['followers_count']>500)&(df['total_likes_given']==0)).astype(int)

# ── GROUP 3: NETWORK INTERACTION ──────────────────────────
df['followers_count']       = df['followers_count'].fillna(0).clip(0,50_000_000)
df['friends_count']         = df['friends_count'].fillna(0).clip(0,5_000_000)
df['follower_friend_ratio'] = (df['followers_count']/(df['friends_count']+1)).clip(0,500)
df['friends_to_followers']  = (df['friends_count']/(df['followers_count']+1)).clip(0,500)
df['likes_to_tweets']       = (df['total_likes_given']/(df['total_tweets']+1)).clip(0,500)
df['network_anomaly_flag']  = ((df['friends_count']>2000)&(df['follower_friend_ratio']<0.1)).astype(int)

# ── GROUP 4: LINGUISTIC CONSISTENCY ──────────────────────
df['has_description']        = df['description'].notna().astype(int)
df['description_length']     = df['description'].fillna('').str.len().clip(0,280)
df['description_word_count'] = df['description'].fillna('').str.split().str.len().clip(0,60)
df['has_location']           = df['location'].notna().astype(int)
df['has_default_pic']        = df['default_profile_image'].astype(int)
df['is_default_profile']     = df['default_profile'].astype(int)
df['is_verified']            = df['verified'].astype(int)
df['has_geo']                = df['geo_enabled'].astype(int)
df['screen_name_length']     = df['screen_name'].fillna('').str.len().clip(0,50)
df['screen_name_digit_ratio']= df['screen_name'].fillna('').apply(lambda x: sum(c.isdigit() for c in x)/(len(x)+1)).clip(0,1)
df['bio_has_url']            = df['description'].fillna('').str.contains('http|www',case=False).astype(int)
df['bio_is_generic']         = ((df['description_length']>0)&(df['description_length']<15)).astype(int)
df['profile_completeness']   = (df['has_description']+df['has_location']+(1-df['has_default_pic'])+(1-df['is_default_profile'])+df['has_geo'])

FEATURES = [
    'tweets_per_day','account_age_days','tweet_density','timing_regularity_score','lifetime_tweet_rate',
    'total_tweets','total_likes_given','likes_per_day','is_high_tweeter','abnormal_tweet_rate',
    'burst_index','burst_flag','sudden_fame','engagement_paradox',
    'followers_count','friends_count','follower_friend_ratio','friends_to_followers','likes_to_tweets','network_anomaly_flag',
    'has_description','description_length','description_word_count','has_location','has_default_pic',
    'is_default_profile','is_verified','has_geo','profile_completeness',
    'screen_name_length','screen_name_digit_ratio','bio_has_url','bio_is_generic'
]

df_clean = df[FEATURES+['label']].copy()
df_clean.replace([np.inf,-np.inf],np.nan,inplace=True)
df_clean.fillna(0,inplace=True)
df_clean['label'] = df_clean['label'].astype(int)
df_clean.to_csv("data/processed_dataset.csv",index=False)

print(f"✅ Done! {len(df_clean):,} records · {len(FEATURES)} features")
print(f"   Bots: {df_clean['label'].sum():,} · Humans: {(df_clean['label']==0).sum():,}")
