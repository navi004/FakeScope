"""
app.py  —  FakeScope Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, json, os

st.set_page_config(page_title="FakeScope", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

:root {
  --bg:        #060810;
  --bg2:       #0B0E1A;
  --bg3:       #0F1221;
  --card:      #111525;
  --card2:     #161A2E;
  --border:    #1E2440;
  --border2:   #252B4A;
  --red:       #FF2D55;
  --red-dim:   rgba(255,45,85,0.12);
  --red-glow:  rgba(255,45,85,0.35);
  --teal:      #00F5C4;
  --teal-dim:  rgba(0,245,196,0.10);
  --teal-glow: rgba(0,245,196,0.30);
  --amber:     #FFB800;
  --amber-dim: rgba(255,184,0,0.10);
  --blue:      #4D9FFF;
  --blue-dim:  rgba(77,159,255,0.10);
  --purple:    #B06FFF;
  --grey:      #5A6080;
  --grey2:     #8892B0;
  --white:     #E8EEFF;
  --mono:      'Share Tech Mono', monospace;
  --display:   'Orbitron', sans-serif;
  --body:      'Rajdhani', sans-serif;
}

html, body, [class*="css"] { font-family: var(--body); color: var(--white); }
.stApp { background: var(--bg); }

section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border2) !important;
}

.main .block-container { background: var(--bg); padding-top: 2rem !important; }

/* PAGE TITLE */
.page-title {
  font-family: var(--display);
  font-size: 26px; font-weight: 900;
  letter-spacing: 3px; text-transform: uppercase;
  color: var(--white);
  border-left: 3px solid var(--teal);
  padding-left: 16px; margin-bottom: 4px;
}
.page-sub {
  font-family: var(--mono);
  font-size: 10px; color: var(--grey);
  padding-left: 19px; letter-spacing: 2px; margin-bottom: 24px;
}

/* SECTION LABELS */
.sec-label {
  font-family: var(--mono);
  font-size: 10px; color: var(--teal);
  letter-spacing: 3px; text-transform: uppercase;
  border-bottom: 1px solid var(--border2);
  padding-bottom: 6px; margin: 24px 0 14px 0;
}

/* METRIC CARDS */
.mcard {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 4px; padding: 20px 16px;
  text-align: center; position: relative; overflow: hidden;
}
.mcard::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
}
.mcard-label {
  font-family: var(--mono); font-size: 9px; color: var(--grey);
  letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 10px;
}
.mcard-value { font-family: var(--display); font-size: 32px; font-weight: 700; line-height: 1; }

.col-teal   { color: var(--teal);   }
.col-red    { color: var(--red);    }
.col-amber  { color: var(--amber);  }
.col-blue   { color: var(--blue);   }
.col-purple { color: var(--purple); }
.col-grey   { color: var(--grey2);  }

/* VERDICT */
.verdict { border-radius: 4px; padding: 24px 32px; text-align: center; margin: 20px 0; }
.verdict-text { font-family: var(--display); font-size: 24px; font-weight: 900; letter-spacing: 4px; text-transform: uppercase; }
.verdict-sub  { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; margin-top: 6px; opacity: 0.7; }
.v-bot        { background: var(--red-dim);   border: 1px solid var(--red);   box-shadow: 0 0 30px var(--red-glow); }
.v-bot .verdict-text { color: var(--red); }
.v-organic    { background: var(--teal-dim);  border: 1px solid var(--teal);  box-shadow: 0 0 30px var(--teal-glow); }
.v-organic .verdict-text { color: var(--teal); }
.v-suspicious { background: var(--amber-dim); border: 1px solid var(--amber); box-shadow: 0 0 20px rgba(255,184,0,0.2); }
.v-suspicious .verdict-text { color: var(--amber); }

/* GAUGE BAR */
.gauge-track { background: var(--border); border-radius: 2px; height: 6px; overflow: hidden; margin: 6px 0; }
.gauge-fill-red   { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #FF2D55, #FF6B7A); }
.gauge-fill-teal  { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #00C49A, #00F5C4); }
.gauge-fill-amber { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #CC9200, #FFB800); }

/* ANOMALY CARDS */
.anom-card {
  background: var(--card2); border: 1px solid var(--border2);
  border-left: 3px solid var(--red); border-radius: 4px;
  padding: 14px 16px; margin: 8px 0;
  font-size: 14px; color: var(--grey2); line-height: 1.6;
}
.anom-card.human { border-left-color: var(--teal); }
.anom-group { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; margin-bottom: 4px; }
.anom-group.t { color: var(--red); }
.anom-group.h { color: var(--teal); }

/* TAGS */
.tag {
  display: inline-block; border-radius: 2px; padding: 3px 10px;
  font-family: var(--mono); font-size: 10px; letter-spacing: 1px;
  margin: 3px; border: 1px solid;
}
.tag-r { border-color: var(--red);    color: var(--red);    background: var(--red-dim); }
.tag-g { border-color: var(--teal);   color: var(--teal);   background: var(--teal-dim); }
.tag-y { border-color: var(--amber);  color: var(--amber);  background: var(--amber-dim); }
.tag-p { border-color: var(--purple); color: var(--purple); background: rgba(176,111,255,0.1); }

/* INSIGHT CARDS */
.insight-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 4px; padding: 18px 16px; min-height: 130px;
}
.insight-name { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
.insight-text { font-family: var(--body); font-size: 13px; color: var(--grey2); line-height: 1.65; }

/* SIDEBAR */
.sidebar-logo { font-family: var(--display); font-size: 20px; font-weight: 900; letter-spacing: 4px; color: var(--teal); text-align: center; padding: 8px 0 4px; }
.sidebar-tagline { font-family: var(--mono); font-size: 9px; color: var(--grey); text-align: center; letter-spacing: 2px; margin-bottom: 16px; }
.sidebar-badge {
  background: var(--card2); border: 1px solid var(--border2);
  border-radius: 4px; padding: 12px 14px;
  font-family: var(--mono); font-size: 10px; color: var(--grey2); line-height: 2; margin-top: 16px;
}
.sidebar-badge .hi { color: var(--teal); }

/* INPUTS */
.stSlider > div > div { background: var(--border) !important; }
.stSlider > div > div > div { background: var(--teal) !important; }
.stSelectbox label, .stSlider label { font-family: var(--mono) !important; font-size: 11px !important; color: var(--grey2) !important; letter-spacing: 1px !important; }

/* BUTTON */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00C49A, #00F5C4) !important;
  color: #060810 !important; font-family: var(--display) !important;
  font-weight: 700 !important; letter-spacing: 3px !important;
  text-transform: uppercase !important; border: none !important;
  border-radius: 4px !important; padding: 14px 0 !important; font-size: 13px !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] { background: var(--card) !important; border-bottom: 1px solid var(--border2) !important; }
.stTabs [data-baseweb="tab"] { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 1.5px !important; color: var(--grey) !important; background: transparent !important; border: none !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; }

.stRadio label { font-family: var(--mono) !important; font-size: 11px !important; color: var(--grey2) !important; letter-spacing: 1px !important; }
h1,h2,h3 { font-family: var(--display) !important; }
hr { border-color: var(--border2) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/rf_model.pkl")
    features = pd.read_csv("models/features.csv", header=None)[0].tolist()
    feat_imp = pd.read_csv("models/feature_importance.csv", index_col=0)['importance']
    return model, features, feat_imp

@st.cache_data
def load_metrics():
    with open("outputs/metrics.json") as f: return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed_dataset.csv")

# ── Feature metadata ────────────────────────────────────────────────────────────
FEATURE_META = {
    'tweets_per_day'           : ('⏱️ TIMING',     'Avg tweets per day',                 'high', 'tag-r'),
    'timing_regularity_score'  : ('⏱️ TIMING',     'Posting regularity score',            'high', 'tag-r'),
    'lifetime_tweet_rate'      : ('⏱️ TIMING',     'Lifetime tweet rate (tweets/hr)',     'high', 'tag-r'),
    'tweet_density'            : ('⏱️ TIMING',     'Tweet density (tweets/day of life)',  'high', 'tag-r'),
    'account_age_days'         : ('⏱️ TIMING',     'Account age in days',                'low',  'tag-r'),
    'burst_index'              : ('💥 BURST',      'Burst activity index',                'high', 'tag-y'),
    'burst_flag'               : ('💥 BURST',      'Burst flag triggered',                'high', 'tag-y'),
    'sudden_fame'              : ('💥 BURST',      'Sudden follower spike',               'high', 'tag-y'),
    'engagement_paradox'       : ('💥 BURST',      'Engagement paradox',                 'high', 'tag-y'),
    'is_high_tweeter'          : ('💥 BURST',      'High tweeter (>50/day)',              'high', 'tag-y'),
    'abnormal_tweet_rate'      : ('💥 BURST',      'Abnormal tweet rate (>100/day)',      'high', 'tag-y'),
    'total_tweets'             : ('💥 BURST',      'Total tweets ever posted',            'high', 'tag-y'),
    'total_likes_given'        : ('💥 BURST',      'Total likes given',                  'low',  'tag-y'),
    'likes_per_day'            : ('💥 BURST',      'Likes given per day',                'low',  'tag-y'),
    'follower_friend_ratio'    : ('🌐 NETWORK',    'Followers ÷ Following ratio',         'low',  'tag-g'),
    'friends_to_followers'     : ('🌐 NETWORK',    'Following ÷ Followers ratio',         'high', 'tag-g'),
    'followers_count'          : ('🌐 NETWORK',    'Total followers',                     'high', 'tag-g'),
    'friends_count'            : ('🌐 NETWORK',    'Total following',                     'high', 'tag-g'),
    'likes_to_tweets'          : ('🌐 NETWORK',    'Likes given per tweet',              'low',  'tag-g'),
    'network_anomaly_flag'     : ('🌐 NETWORK',    'Network anomaly flag',                'high', 'tag-g'),
    'screen_name_digit_ratio'  : ('🗣️ LINGUISTIC', 'Username digit ratio',                'high', 'tag-p'),
    'screen_name_length'       : ('🗣️ LINGUISTIC', 'Username length',                     'high', 'tag-p'),
    'has_default_pic'          : ('🗣️ LINGUISTIC', 'Default profile picture',             'high', 'tag-p'),
    'is_default_profile'       : ('🗣️ LINGUISTIC', 'Default profile theme',               'high', 'tag-p'),
    'bio_is_generic'           : ('🗣️ LINGUISTIC', 'Generic/empty bio',                  'high', 'tag-p'),
    'bio_has_url'              : ('🗣️ LINGUISTIC', 'Bio contains URL',                    'high', 'tag-p'),
    'has_description'          : ('🗣️ LINGUISTIC', 'Has bio/description',                'low',  'tag-p'),
    'description_length'       : ('🗣️ LINGUISTIC', 'Bio length (chars)',                 'low',  'tag-p'),
    'description_word_count'   : ('🗣️ LINGUISTIC', 'Bio word count',                     'low',  'tag-p'),
    'has_location'             : ('🗣️ LINGUISTIC', 'Has location set',                   'low',  'tag-p'),
    'has_geo'                  : ('🗣️ LINGUISTIC', 'Geo-tagging enabled',                'low',  'tag-p'),
    'is_verified'              : ('🗣️ LINGUISTIC', 'Verified account',                   'low',  'tag-p'),
    'profile_completeness'     : ('🗣️ LINGUISTIC', 'Profile completeness (0-5)',          'low',  'tag-p'),
}

def get_per_account_contributions(row_dict, model, features):
    X_in = pd.DataFrame([row_dict])[features]
    contributions = np.zeros(len(features))
    for tree in model.estimators_:
        node_ids  = tree.decision_path(X_in).indices
        tree_     = tree.tree_
        prev_prob = tree_.value[0][0][1] / tree_.value[0][0].sum()
        for node_id in node_ids[1:]:
            cur_prob   = tree_.value[node_id][0][1] / tree_.value[node_id][0].sum()
            split_feat = tree_.feature[node_id]
            if split_feat >= 0:
                contributions[split_feat] += (cur_prob - prev_prob)
            prev_prob = cur_prob
    contributions /= len(model.estimators_)
    return pd.Series(contributions, index=features)

def explain_account(row_dict, model, features):
    X_in       = pd.DataFrame([row_dict])[features]
    prob_bot   = model.predict_proba(X_in)[0][1]
    bot_score  = round(prob_bot * 100, 1)
    auth_score = round(100 - bot_score, 1)

    if bot_score >= 70:   verdict = ("🤖 BOT DETECTED",        "v-bot",        "THREAT LEVEL: HIGH")
    elif bot_score >= 40: verdict = ("⚠️  SUSPICIOUS ACCOUNT", "v-suspicious", "THREAT LEVEL: MEDIUM")
    else:                 verdict = ("✅ ORGANIC USER",         "v-organic",    "THREAT LEVEL: LOW")

    contribs  = get_per_account_contributions(row_dict, model, features)
    top_bot   = contribs.sort_values(ascending=False).head(5)
    top_human = contribs.sort_values(ascending=True).head(3)

    anomalies = []
    for feat, contrib in top_bot.items():
        if contrib <= 0: continue
        val  = row_dict.get(feat, 0)
        meta = FEATURE_META.get(feat, ('❓','Unknown','high','tag-r'))
        group, readable, direction, tag_cls = meta
        val_str   = f'{val:.2f}' if isinstance(val, float) and val != int(val) else str(int(val))
        suspicion = 'HIGH = bot signal' if direction == 'high' else 'LOW = bot signal'
        anomalies.append((group, readable, val_str, suspicion, round(contrib*100, 2), tag_cls))

    human_signals = []
    for feat, contrib in top_human.items():
        if contrib >= 0: continue
        val  = row_dict.get(feat, 0)
        meta = FEATURE_META.get(feat, ('❓','Unknown','low','tag-g'))
        group, readable, _, _ = meta
        val_str = f'{val:.2f}' if isinstance(val, float) and val != int(val) else str(int(val))
        human_signals.append((group, readable, val_str, round(abs(contrib)*100, 2)))

    if not anomalies:
        anomalies = [('✅ CLEAN', 'No significant bot signals detected', '—', '—', 0, 'tag-g')]

    return bot_score, auth_score, verdict, anomalies, human_signals


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>FAKESCOPE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>BEHAVIOURAL BOT DETECTION ENGINE</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio("NAV", [
        "🎯 Live Analyzer",
        "📊 Model Performance",
        "🔬 Dataset Explorer",
        "💡 Behavioural Insights"
    ], label_visibility="collapsed")

    st.markdown("""
    <div class='sidebar-badge'>
      <span class='hi'>▸ SYSTEM</span><br/>
      Hackathon: Behavioural Analytics<br/>
      Problem Statement: PS-3<br/>
      <br/>
      <span class='hi'>▸ DATASET</span><br/>
      airt-ml/twitter-human-bots<br/>
      37,400 real accounts<br/>
      License: CC-BY-SA 3.0<br/>
      <br/>
      <span class='hi'>▸ OUTPUTS</span><br/>
      ● Bot Probability<br/>
      ● Authenticity Score<br/>
      ● Anomaly Explanation<br/>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — Live Analyzer
# ══════════════════════════════════════════════════════════════════
if page == "🎯 Live Analyzer":
    st.markdown("<div class='page-title'>Live Account Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>INPUT BEHAVIOURAL METRICS → RECEIVE THREAT ASSESSMENT</div>", unsafe_allow_html=True)

    try: model, feature_names, feat_imp = load_model()
    except: st.error("⚠️ Run the Colab notebook first to generate models/"); st.stop()

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("<div class='sec-label'>⏱ Timing & Activity</div>", unsafe_allow_html=True)
        tweets_per_day    = st.slider("Avg Tweets per Day",   0.0, 300.0, 5.0, step=0.5)
        account_age_days  = st.slider("Account Age (days)",   0, 5000, 500)
        total_tweets      = st.slider("Total Tweets",         0, 500000, 2000, step=100)
        total_likes_given = st.slider("Total Likes Given",    0, 500000, 1000, step=100)

    with c2:
        st.markdown("<div class='sec-label'>🌐 Network Metrics</div>", unsafe_allow_html=True)
        followers_count = st.slider("Followers",  0, 1000000, 1200, step=100)
        friends_count   = st.slider("Following",  0, 500000,  800,  step=100)

        st.markdown("<div class='sec-label'>🗣 Profile Signals</div>", unsafe_allow_html=True)
        has_description    = st.selectbox("Has Bio?",          [1, 0], format_func=lambda x: "Yes" if x else "No")
        description_length = st.slider("Bio Length (chars)", 0, 280, 80) if has_description else 0
        has_location       = st.selectbox("Has Location?",    [1, 0], format_func=lambda x: "Yes" if x else "No")
        has_default_pic    = st.selectbox("Default Picture?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_default_profile = st.selectbox("Default Theme?",   [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_verified        = st.selectbox("Verified?",        [0, 1], format_func=lambda x: "Yes" if x else "No")
        has_geo            = st.selectbox("Geo-enabled?",     [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c3:
        st.markdown("<div class='sec-label'>🔤 Username Signals</div>", unsafe_allow_html=True)
        screen_name_length      = st.slider("Username Length (chars)",    1,   50,  10)
        screen_name_digit_ratio = st.slider("Username Digit Ratio (0–1)", 0.0, 1.0, 0.05, step=0.01,
                                            help="e.g. 'user_48291847' = 0.53. High ratio = bot signal.")
        bio_has_url = st.selectbox("Bio contains URL?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        # Live derived indicators
        st.markdown("<div class='sec-label' style='margin-top:28px;'>⚡ Derived Indicators</div>", unsafe_allow_html=True)
        _trs = round(tweets_per_day / (np.log1p(account_age_days) + 1), 2)
        _bi  = round(total_tweets / (account_age_days + 1), 2)
        _ffr = round(followers_count / (friends_count + 1), 2)
        _pc  = has_description + has_location + (1-has_default_pic) + (1-is_default_profile) + has_geo

        def gauge_color(val, hi_thresh, lo_thresh):
            return "red" if val >= hi_thresh else ("amber" if val >= lo_thresh else "teal")

        st.markdown(f"""
        <div style='background:var(--card);border:1px solid var(--border);border-radius:4px;padding:14px;'>
          <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
            <span style='font-family:var(--mono);font-size:10px;color:var(--grey);'>TIMING REGULARITY</span>
            <span style='font-family:var(--mono);font-size:12px;color:{"var(--red)" if _trs>5 else ("var(--amber)" if _trs>1 else "var(--teal)")};'>{_trs}</span>
          </div>
          <div class='gauge-track'><div class='gauge-fill-{gauge_color(_trs,5,1)}' style='width:{min(_trs/20*100,100):.0f}%'></div></div>
          <div style='display:flex;justify-content:space-between;margin-bottom:6px;margin-top:10px;'>
            <span style='font-family:var(--mono);font-size:10px;color:var(--grey);'>BURST INDEX</span>
            <span style='font-family:var(--mono);font-size:12px;color:{"var(--red)" if _bi>50 else ("var(--amber)" if _bi>10 else "var(--teal)")};'>{_bi}</span>
          </div>
          <div class='gauge-track'><div class='gauge-fill-{gauge_color(_bi,50,10)}' style='width:{min(_bi/200*100,100):.0f}%'></div></div>
          <div style='display:flex;justify-content:space-between;margin-bottom:6px;margin-top:10px;'>
            <span style='font-family:var(--mono);font-size:10px;color:var(--grey);'>FOLLOWER RATIO</span>
            <span style='font-family:var(--mono);font-size:12px;color:{"var(--teal)" if _ffr>0.5 else ("var(--amber)" if _ffr>0.1 else "var(--red)")};'>{_ffr}</span>
          </div>
          <div class='gauge-track'><div class='gauge-fill-teal' style='width:{min(_ffr/5*100,100):.0f}%'></div></div>
          <div style='display:flex;justify-content:space-between;margin-bottom:6px;margin-top:10px;'>
            <span style='font-family:var(--mono);font-size:10px;color:var(--grey);'>PROFILE COMPLETENESS</span>
            <span style='font-family:var(--mono);font-size:12px;color:{"var(--teal)" if _pc>=4 else ("var(--amber)" if _pc>=2 else "var(--red)")};'>{int(_pc)}/5</span>
          </div>
          <div class='gauge-track'><div class='gauge-fill-{"teal" if _pc>=4 else ("amber" if _pc>=2 else "red")}' style='width:{_pc/5*100:.0f}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

    # Derived features
    tweet_density           = total_tweets / (account_age_days + 1)
    likes_per_day           = total_likes_given / (account_age_days + 1)
    timing_regularity_score = tweets_per_day / (np.log1p(account_age_days) + 1)
    lifetime_tweet_rate     = total_tweets / (account_age_days * 24 + 1)
    follower_friend_ratio   = followers_count / (friends_count + 1)
    friends_to_followers    = friends_count / (followers_count + 1)
    likes_to_tweets         = total_likes_given / (total_tweets + 1)
    burst_index             = total_tweets / (account_age_days + 1)
    description_word_count  = len(str(description_length).split()) if has_description else 0
    profile_completeness    = has_description + has_location + (1-has_default_pic) + (1-is_default_profile) + has_geo

    feat_dict = {
        'tweets_per_day': tweets_per_day, 'account_age_days': account_age_days,
        'tweet_density': tweet_density, 'timing_regularity_score': timing_regularity_score,
        'lifetime_tweet_rate': lifetime_tweet_rate,
        'total_tweets': total_tweets, 'total_likes_given': total_likes_given,
        'likes_per_day': likes_per_day,
        'is_high_tweeter': int(tweets_per_day > 50), 'abnormal_tweet_rate': int(tweets_per_day > 100),
        'burst_index': burst_index, 'burst_flag': int(burst_index > 20),
        'sudden_fame': int(account_age_days < 30 and followers_count > 1000),
        'engagement_paradox': int(followers_count > 500 and total_likes_given == 0),
        'followers_count': followers_count, 'friends_count': friends_count,
        'follower_friend_ratio': follower_friend_ratio, 'friends_to_followers': friends_to_followers,
        'likes_to_tweets': likes_to_tweets,
        'network_anomaly_flag': int(friends_count > 2000 and follower_friend_ratio < 0.1),
        'has_description': has_description, 'description_length': description_length,
        'description_word_count': description_word_count,
        'has_location': has_location, 'has_default_pic': has_default_pic,
        'is_default_profile': is_default_profile, 'is_verified': is_verified,
        'has_geo': has_geo, 'profile_completeness': profile_completeness,
        'screen_name_length': screen_name_length,
        'screen_name_digit_ratio': screen_name_digit_ratio,
        'bio_has_url': bio_has_url,
        'bio_is_generic': int(has_description and description_length < 15),
    }

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("🔍  RUN THREAT ANALYSIS", use_container_width=True, type="primary")

    if analyze:
        bot_score, auth_score, verdict_data, anomalies, human_signals = explain_account(
            feat_dict, model, feature_names
        )
        v_text, v_cls, v_sub = verdict_data

        st.markdown(f"""
        <div class='verdict {v_cls}'>
          <div class='verdict-text'>{v_text}</div>
          <div class='verdict-sub'>{v_sub}</div>
        </div>""", unsafe_allow_html=True)

        sc  = "col-red"   if bot_score >= 70  else ("col-amber" if bot_score >= 40  else "col-teal")
        ac  = "col-teal"  if auth_score >= 60 else ("col-amber" if auth_score >= 30 else "col-red")
        pc_c= "col-teal"  if profile_completeness >= 4 else ("col-amber" if profile_completeness >= 2 else "col-red")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
            <div class='mcard'>
              <div class='mcard-label'>Bot Probability</div>
              <div class='mcard-value {sc}'>{bot_score}%</div>
              <div class='gauge-track' style='margin-top:12px;'>
                <div class='gauge-fill-{"red" if bot_score>=70 else ("amber" if bot_score>=40 else "teal")}' style='width:{bot_score}%'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class='mcard'>
              <div class='mcard-label'>Authenticity Score</div>
              <div class='mcard-value {ac}'>{auth_score}%</div>
              <div class='gauge-track' style='margin-top:12px;'>
                <div class='gauge-fill-{"teal" if auth_score>=60 else ("amber" if auth_score>=30 else "red")}' style='width:{auth_score}%'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class='mcard'>
              <div class='mcard-label'>Profile Completeness</div>
              <div class='mcard-value {pc_c}'>{int(profile_completeness)}<span style='font-size:16px;color:var(--grey);'>/5</span></div>
              <div class='gauge-track' style='margin-top:12px;'>
                <div class='gauge-fill-{"teal" if profile_completeness>=4 else ("amber" if profile_completeness>=2 else "red")}' style='width:{profile_completeness/5*100:.0f}%'></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-label' style='margin-top:28px;'>⚡ Behavioural Anomaly Explanation — Model-Driven</div>", unsafe_allow_html=True)
        for group, readable, val_str, suspicion, push, tag_cls in anomalies:
            st.markdown(f"""
            <div class='anom-card'>
              <div class='anom-group t'>{group}</div>
              <span class='tag {tag_cls}'>{readable}</span>&nbsp;
              <span style='font-family:var(--mono);font-size:12px;color:var(--white);'>{val_str}</span>&nbsp;·&nbsp;
              <span style='font-family:var(--mono);font-size:11px;color:var(--grey2);'>{suspicion}</span>&nbsp;·&nbsp;
              <span style='font-family:var(--mono);font-size:11px;color:var(--red);'>+{push}% → bot</span>
            </div>""", unsafe_allow_html=True)

        if human_signals:
            st.markdown("<div class='sec-label' style='margin-top:20px;'>✅ Organic Signals — Pulling Toward Human</div>", unsafe_allow_html=True)
            for group, readable, val_str, pull in human_signals:
                st.markdown(f"""
                <div class='anom-card human'>
                  <div class='anom-group h'>{group}</div>
                  <span class='tag tag-g'>{readable}</span>&nbsp;
                  <span style='font-family:var(--mono);font-size:12px;color:var(--white);'>{val_str}</span>&nbsp;·&nbsp;
                  <span style='font-family:var(--mono);font-size:11px;color:var(--teal);'>−{pull}% → human</span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("<div class='page-title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>RANDOM FOREST · 37,400 REAL ACCOUNTS · 33 BEHAVIOURAL FEATURES</div>", unsafe_allow_html=True)

    try: m = load_metrics()
    except: st.error("Run the Colab notebook first."); st.stop()

    cols = st.columns(5)
    labels = [("Accuracy","accuracy","col-teal"),("Precision","precision","col-blue"),
              ("Recall","recall","col-amber"),("F1 Score","f1_score","col-purple"),("ROC AUC","roc_auc","col-teal")]
    for col, (lbl, key, cls) in zip(cols, labels):
        with col:
            st.markdown(f"""
            <div class='mcard'>
              <div class='mcard-label'>{lbl}</div>
              <div class='mcard-value {cls}' style='font-size:26px;'>{m[key]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='mcard' style='margin-top:12px;'>
      <div class='mcard-label'>5-Fold Cross-Validation AUC</div>
      <div class='mcard-value col-teal' style='font-size:22px;'>
        {m.get('cv_auc_mean','?')} <span style='font-size:14px;color:var(--grey);'>± {m.get('cv_auc_std','?')}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Visualizations</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Confusion Matrix","ROC Curve","Feature Importance","Group Importance","Distributions"])
    imgs = ["outputs/confusion_matrix.png","outputs/roc_curve.png","outputs/feature_importance.png",
            "outputs/group_importance.png","outputs/feature_distributions.png"]
    for tab, path in zip(tabs, imgs):
        with tab:
            if os.path.exists(path): st.image(path, use_container_width=True)
            else: st.info("Run the Colab notebook to generate this plot.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Dataset Explorer":
    st.markdown("<div class='page-title'>Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>airt-ml/twitter-human-bots · HUGGINGFACE · CC-BY-SA 3.0</div>", unsafe_allow_html=True)

    try: df = load_dataset()
    except: st.error("Run the Colab notebook first."); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, cls) in zip([c1,c2,c3,c4],[
        ("Total Records",  len(df),                "col-amber"),
        ("Human Accounts",(df['label']==0).sum(),  "col-teal"),
        ("Bot Accounts",  (df['label']==1).sum(),  "col-red"),
        ("Features",      len(df.columns)-1,        "col-purple"),
    ]):
        with col:
            st.markdown(f"""
            <div class='mcard'>
              <div class='mcard-label'>{lbl}</div>
              <div class='mcard-value {cls}' style='font-size:26px;'>{val:,}</div>
            </div>""", unsafe_allow_html=True)

    DARK_BG='#060810'; CARD_BG='#111525'; BORDER='#1E2440'; GREEN='#00F5C4'; RED='#FF2D55'; GREY='#5A6080'

    st.markdown("<div class='sec-label'>Feature Distribution — Bot vs Human</div>", unsafe_allow_html=True)
    col_g, col_f = st.columns([1, 3])
    with col_g:
        group = st.radio("Group", ["⏱️ Timing","💥 Burst","🌐 Network","🗣️ Linguistic"], label_visibility="collapsed")
    group_map = {
        "⏱️ Timing":    ['tweets_per_day','timing_regularity_score','lifetime_tweet_rate','tweet_density'],
        "💥 Burst":     ['burst_index','sudden_fame','engagement_paradox','abnormal_tweet_rate'],
        "🌐 Network":   ['follower_friend_ratio','friends_to_followers','network_anomaly_flag','likes_to_tweets'],
        "🗣️ Linguistic":['screen_name_digit_ratio','profile_completeness','description_length','bio_is_generic'],
    }
    with col_f:
        feat = st.selectbox("Feature", group_map[group], label_visibility="collapsed")

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.hist(df[df['label']==0][feat], bins=50, alpha=0.75, color=GREEN, density=True, label='Human')
    ax.hist(df[df['label']==1][feat], bins=50, alpha=0.75, color=RED,   density=True, label='Bot')
    ax.set_xlabel(feat, color=GREY, fontsize=11, fontfamily='monospace')
    ax.set_ylabel('Density', color=GREY, fontsize=11)
    ax.tick_params(colors=GREY)
    ax.legend(fontsize=11, framealpha=0.2, facecolor=CARD_BG, labelcolor='white')
    ax.grid(alpha=0.12, color=BORDER)
    st.pyplot(fig)

    st.markdown("<div class='sec-label'>Raw Data Preview</div>", unsafe_allow_html=True)
    filt = st.selectbox("Filter", ["All","Humans Only","Bots Only"], label_visibility="collapsed")
    dshow = df[df['label']==0] if filt=="Humans Only" else (df[df['label']==1] if filt=="Bots Only" else df)
    st.dataframe(dshow.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — Behavioural Insights
# ══════════════════════════════════════════════════════════════════
elif page == "💡 Behavioural Insights":
    st.markdown("<div class='page-title'>Behavioural Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>ALL 4 PS-3 INDICATOR GROUPS · DERIVED FROM 37,400 REAL ACCOUNTS</div>", unsafe_allow_html=True)

    groups = [
        ("⏱️ Timing Regularity", "#FF2D55", [
            ("timing_regularity_score", "Bots post at superhuman rates relative to account age. A score above 10 is a near-certain bot signal."),
            ("tweets_per_day",          "Organic users average 1–10 tweets/day. Anything above 50 is highly suspicious; above 100 is almost always a bot."),
            ("lifetime_tweet_rate",     "Measures what fraction of the account's waking hours were spent tweeting. Bots approach 24/7 saturation."),
        ]),
        ("💥 Engagement Burst Patterns", "#FFB800", [
            ("burst_index",        "A high burst index means massive volume in short time — characteristic of coordinated bot campaigns."),
            ("sudden_fame",        "A brand-new account with thousands of followers has bought or manufactured its audience — a classic bot farm signal."),
            ("engagement_paradox", "An account with many followers that has never liked a single post reveals its followers are also fake."),
        ]),
        ("🌐 Network Interaction Patterns", "#00F5C4", [
            ("follower_friend_ratio", "Bots follow thousands but attract few genuine followers back. Extreme low ratios (<0.1) strongly indicate bot behaviour."),
            ("network_anomaly_flag",  "Flagged when an account follows >2000 people but has almost no followers — a coordinated follow-spam pattern."),
            ("likes_to_tweets",       "Real users like posts — this reciprocity ratio is near-zero for bots that only broadcast, never engage."),
        ]),
        ("🗣️ Linguistic Consistency", "#B06FFF", [
            ("screen_name_digit_ratio","Bot names like 'user_48291847' have high digit ratios. A ratio >0.4 is a strong signal."),
            ("profile_completeness",   "Bots skip profile setup — no bio, no location, default picture, default theme. Score <2 is highly suspicious."),
            ("bio_is_generic",         "A bio shorter than 15 characters is often a bot placeholder. Real users write meaningful descriptions."),
        ]),
    ]

    for group_name, color, features_info in groups:
        st.markdown(f"""
        <div style='border-left:3px solid {color};padding:2px 0 2px 16px;margin:28px 0 14px;'>
          <span style='font-family:var(--display);font-weight:700;font-size:15px;color:{color};letter-spacing:2px;'>{group_name}</span>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(3, gap="medium")
        for col, (fname, explanation) in zip(cols, features_info):
            with col:
                st.markdown(f"""
                <div class='insight-card'>
                  <div class='insight-name' style='color:{color};'>{fname.replace('_',' ').upper()}</div>
                  <div class='insight-text'>{explanation}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:32px;'>Feature Importance by Group</div>", unsafe_allow_html=True)
    if os.path.exists("outputs/group_importance.png"):
        st.image("outputs/group_importance.png", use_container_width=True)

    st.markdown("<div class='sec-label'>Top 20 Feature Importance</div>", unsafe_allow_html=True)
    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png", use_container_width=True)