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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0D0F14;}
.metric-card{background:linear-gradient(135deg,#1A1D26,#12141C);border:1px solid #2A2D3A;border-radius:16px;padding:24px 20px;text-align:center;margin:6px 0;}
.metric-label{font-family:'Space Mono',monospace;font-size:11px;color:#8B8FA8;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}
.metric-value{font-size:36px;font-weight:700;line-height:1;}
.c-green{color:#4ECCA3;}.c-red{color:#FF4757;}.c-yellow{color:#FFE66D;}.c-purple{color:#A78BFA;}
.verdict-box{border-radius:16px;padding:28px;text-align:center;font-family:'Space Mono',monospace;font-size:22px;font-weight:700;letter-spacing:1px;margin:16px 0;}
.v-organic{background:rgba(78,204,163,0.12);border:2px solid #4ECCA3;color:#4ECCA3;}
.v-bot{background:rgba(255,71,87,0.12);border:2px solid #FF4757;color:#FF4757;}
.v-suspicious{background:rgba(255,230,109,0.12);border:2px solid #FFE66D;color:#FFE66D;}
.sec{font-family:'Space Mono',monospace;font-size:11px;color:#8B8FA8;text-transform:uppercase;letter-spacing:3px;margin:20px 0 10px 0;padding-bottom:6px;border-bottom:1px solid #2A2D3A;}
.tag{display:inline-block;background:#1A1D26;border:1px solid #2A2D3A;border-radius:20px;padding:6px 14px;font-size:12px;color:#8B8FA8;margin:4px;}
.tag-r{border-color:#FF4757;color:#FF4757;}.tag-g{border-color:#4ECCA3;color:#4ECCA3;}
.tag-y{border-color:#FFE66D;color:#FFE66D;}.tag-p{border-color:#A78BFA;color:#A78BFA;}
.anomaly-box{background:#1A1D26;border:1px solid #FF4757;border-radius:12px;padding:16px;margin:8px 0;font-size:13px;color:#C0C4D4;line-height:1.8;}
h1,h2,h3{font-family:'Space Mono',monospace!important;}
</style>""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
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

# ── Feature metadata for explanation labels ───────────────────────────────────
FEATURE_META = {
    'tweets_per_day'           : ('⏱️ Timing',     'Avg tweets per day',                   'high', 'tag-r'),
    'timing_regularity_score'  : ('⏱️ Timing',     'Posting regularity score',              'high', 'tag-r'),
    'lifetime_tweet_rate'      : ('⏱️ Timing',     'Lifetime tweet rate (tweets/hr)',       'high', 'tag-r'),
    'tweet_density'            : ('⏱️ Timing',     'Tweet density (tweets/day of life)',    'high', 'tag-r'),
    'account_age_days'         : ('⏱️ Timing',     'Account age in days',                  'low',  'tag-r'),
    'burst_index'              : ('💥 Burst',      'Burst activity index',                  'high', 'tag-y'),
    'burst_flag'               : ('💥 Burst',      'Burst flag triggered',                  'high', 'tag-y'),
    'sudden_fame'              : ('💥 Burst',      'Sudden follower spike',                 'high', 'tag-y'),
    'engagement_paradox'       : ('💥 Burst',      'Engagement paradox',                   'high', 'tag-y'),
    'is_high_tweeter'          : ('💥 Burst',      'High tweeter (>50/day)',                'high', 'tag-y'),
    'abnormal_tweet_rate'      : ('💥 Burst',      'Abnormal tweet rate (>100/day)',        'high', 'tag-y'),
    'total_tweets'             : ('💥 Burst',      'Total tweets ever posted',              'high', 'tag-y'),
    'total_likes_given'        : ('💥 Burst',      'Total likes given',                    'low',  'tag-y'),
    'likes_per_day'            : ('💥 Burst',      'Likes given per day',                  'low',  'tag-y'),
    'follower_friend_ratio'    : ('🌐 Network',    'Followers ÷ Following ratio',           'low',  'tag-g'),
    'friends_to_followers'     : ('🌐 Network',    'Following ÷ Followers ratio',           'high', 'tag-g'),
    'followers_count'          : ('🌐 Network',    'Total followers',                       'high', 'tag-g'),
    'friends_count'            : ('🌐 Network',    'Total following',                       'high', 'tag-g'),
    'likes_to_tweets'          : ('🌐 Network',    'Likes given per tweet',                'low',  'tag-g'),
    'network_anomaly_flag'     : ('🌐 Network',    'Network anomaly flag',                  'high', 'tag-g'),
    'screen_name_digit_ratio'  : ('🗣️ Linguistic', 'Username digit ratio',                  'high', 'tag-p'),
    'screen_name_length'       : ('🗣️ Linguistic', 'Username length',                       'high', 'tag-p'),
    'has_default_pic'          : ('🗣️ Linguistic', 'Default profile picture',               'high', 'tag-p'),
    'is_default_profile'       : ('🗣️ Linguistic', 'Default profile theme',                 'high', 'tag-p'),
    'bio_is_generic'           : ('🗣️ Linguistic', 'Generic/empty bio',                    'high', 'tag-p'),
    'bio_has_url'              : ('🗣️ Linguistic', 'Bio contains URL',                      'high', 'tag-p'),
    'has_description'          : ('🗣️ Linguistic', 'Has bio/description',                  'low',  'tag-p'),
    'description_length'       : ('🗣️ Linguistic', 'Bio length (chars)',                   'low',  'tag-p'),
    'description_word_count'   : ('🗣️ Linguistic', 'Bio word count',                       'low',  'tag-p'),
    'has_location'             : ('🗣️ Linguistic', 'Has location set',                     'low',  'tag-p'),
    'has_geo'                  : ('🗣️ Linguistic', 'Geo-tagging enabled',                  'low',  'tag-p'),
    'is_verified'              : ('🗣️ Linguistic', 'Verified account',                     'low',  'tag-p'),
    'profile_completeness'     : ('🗣️ Linguistic', 'Profile completeness (0-5)',            'low',  'tag-p'),
}

def get_per_account_contributions(row_dict, model, features):
    """Model-driven: walks each tree's decision path to find per-account feature contributions."""
    X_in  = pd.DataFrame([row_dict])[features]
    X_arr = np.asarray(X_in)
    x_arr = X_arr[0]
    contributions = np.zeros(len(features))
    for tree in model.estimators_:
        node_ids   = tree.decision_path(X_arr).indices
        tree_      = tree.tree_
        prev_prob  = tree_.value[0][0][1] / tree_.value[0][0].sum()
        for node_id in node_ids[1:]:
            cur_prob  = tree_.value[node_id][0][1] / tree_.value[node_id][0].sum()
            split_feat = tree_.feature[node_id]
            if split_feat >= 0:
                contributions[split_feat] += (cur_prob - prev_prob)
            prev_prob = cur_prob
    contributions /= len(model.estimators_)
    return pd.Series(contributions, index=features)

def explain_account(row_dict, model, features, feat_imp_series=None):
    """
    Returns all 3 PS-required outputs:
    OUTPUT 1 — Bot Probability    : model.predict_proba score (0-100%)
    OUTPUT 2 — Authenticity Score : 100 - Bot Probability
    OUTPUT 3 — Behavioural Anomaly Explanation:
               Per-account model-driven feature contributions —
               what the model actually used for THIS specific account.
    """
    X_in       = pd.DataFrame([row_dict])[features]
    prob_bot   = model.predict_proba(np.asarray(X_in))[0][1]
    bot_score  = round(prob_bot * 100, 1)
    auth_score = round(100 - bot_score, 1)

    if bot_score >= 70:   verdict = ("🤖 BOT DETECTED",       "v-bot")
    elif bot_score >= 40: verdict = ("⚠️ SUSPICIOUS ACCOUNT", "v-suspicious")
    else:                 verdict = ("✅ ORGANIC USER",        "v-organic")

    # Model-driven per-account contributions
    contribs      = get_per_account_contributions(row_dict, model, features)
    top_bot       = contribs.sort_values(ascending=False).head(5)
    top_human     = contribs.sort_values(ascending=True).head(3)

    anomalies = []
    for feat, contrib in top_bot.items():
        if contrib <= 0: continue
        val  = row_dict.get(feat, 0)
        meta = FEATURE_META.get(feat, ('❓','Unknown','high','tag'))
        group, readable, direction, tag_cls = meta
        val_str = f'{val:.2f}' if isinstance(val, float) and val != int(val) else str(int(val))
        suspicion = 'high = bot signal' if direction == 'high' else 'low = bot signal'
        anomalies.append((group, f'{readable} = {val_str}  ({suspicion})  · pushed +{round(contrib*100,2)}% toward bot', tag_cls))

    human_signals = []
    for feat, contrib in top_human.items():
        if contrib >= 0: continue
        val  = row_dict.get(feat, 0)
        meta = FEATURE_META.get(feat, ('❓','Unknown','low','tag'))
        group, readable, _, _ = meta
        val_str = f'{val:.2f}' if isinstance(val, float) and val != int(val) else str(int(val))
        human_signals.append(f'{readable} = {val_str}  (organic signal, pulled {round(abs(contrib)*100,2)}% toward human)')

    if not anomalies:
        anomalies = [('✅ Clean', 'No significant bot signals detected — organic behaviour patterns.', 'tag-g')]

    return bot_score, auth_score, verdict, anomalies, human_signals

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FakeScope")
    st.markdown("<p style='color:#8B8FA8;font-size:13px;'>Behavioural Bot Detection Engine</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Page", [
        "🎯 Live Analyzer",
        "📊 Model Performance",
        "🔬 Dataset Explorer",
        "💡 Behavioural Insights"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""<p style='color:#8B8FA8;font-size:11px;font-family:monospace;'>
        Behavioural Analytics Hackathon<br/>
        Problem Statement 3<br/>
        Dataset: airt-ml/twitter-human-bots<br/>
        License: CC-BY-SA 3.0<br/><br/>
        ✅ Outputs:<br/>
        • Bot Probability<br/>
        • Authenticity Score<br/>
        • Anomaly Explanation
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — Live Analyzer
# ══════════════════════════════════════════════════════════════════
if page == "🎯 Live Analyzer":
    st.markdown("# 🎯 Live Account Analyzer")
    st.markdown("<p style='color:#8B8FA8;'>Enter account behavioural metrics → get Bot Probability, Authenticity Score, and Anomaly Explanation.</p>", unsafe_allow_html=True)

    try: model, feature_names, feat_imp = load_model()
    except: st.error("⚠️ Run the Colab notebook first to generate models/"); st.stop()


    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("<p class='sec'>⏱️ Timing & Activity</p>", unsafe_allow_html=True)
        tweets_per_day    = st.slider("Avg Tweets per Day",    0.0,  300.0,  5.0)
        account_age_days  = st.slider("Account Age (days)",    0,    5000,   500)
        total_tweets      = st.slider("Total Tweets",          0,    500000, 2000)
        total_likes_given = st.slider("Total Likes Given",     0,    500000, 1000)

    with c2:
        st.markdown("<p class='sec'>🌐 Network Metrics</p>", unsafe_allow_html=True)
        followers_count = st.slider("Followers",  0, 1000000, 1200)
        friends_count   = st.slider("Following",  0, 500000,  800)

        st.markdown("<p class='sec'>🗣️ Linguistic / Profile</p>", unsafe_allow_html=True)
        has_description   = st.selectbox("Has Bio?",            [1,0], format_func=lambda x:"Yes" if x else "No")
        description_length= st.slider("Bio Length (chars)", 0, 280, 80) if has_description else 0
        has_location      = st.selectbox("Has Location?",       [1,0], format_func=lambda x:"Yes" if x else "No")
        has_default_pic   = st.selectbox("Default Picture?",    [0,1], format_func=lambda x:"Yes" if x else "No")
        is_default_profile= st.selectbox("Default Theme?",      [0,1], format_func=lambda x:"Yes" if x else "No")
        is_verified       = st.selectbox("Verified?",           [0,1], format_func=lambda x:"Yes" if x else "No")
        has_geo           = st.selectbox("Geo-enabled?",        [0,1], format_func=lambda x:"Yes" if x else "No")

    with c3:
        st.markdown("<p class='sec'>🗣️ Username Signals</p>", unsafe_allow_html=True)
        screen_name_length      = st.slider("Username Length (chars)", 1, 50, 10)
        screen_name_digit_ratio = st.slider("Username Digit Ratio (0–1)", 0.0, 1.0, 0.05,
                                            help="e.g. 'user_48291847' = 0.53. Bots often have high ratios.")
        bio_has_url   = st.selectbox("Bio contains URL?",   [0,1], format_func=lambda x:"Yes" if x else "No")

    # Derived features
    tweet_density          = (total_tweets / (account_age_days + 1))
    likes_per_day          = (total_likes_given / (account_age_days + 1))
    timing_regularity_score= (tweets_per_day / (np.log1p(account_age_days) + 1))
    lifetime_tweet_rate    = (total_tweets / (account_age_days * 24 + 1))
    follower_friend_ratio  = (followers_count / (friends_count + 1))
    friends_to_followers   = (friends_count / (followers_count + 1))
    likes_to_tweets        = (total_likes_given / (total_tweets + 1))
    burst_index            = (total_tweets / (account_age_days + 1))
    description_word_count = len(str(description_length).split()) if has_description else 0
    profile_completeness   = has_description + has_location + (1-has_default_pic) + (1-is_default_profile) + has_geo

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

    st.markdown("---")
    if st.button("🔍 Analyze Account", use_container_width=True, type="primary"):
        bot_score, auth_score, verdict_data, anomalies, human_signals = explain_account(
            feat_dict, model, feature_names
        )
        v_text, v_cls = verdict_data

        # Verdict
        st.markdown(f"<div class='verdict-box {v_cls}'>{v_text}</div>", unsafe_allow_html=True)

        # Output 1 & 2 — Scores
        mc1, mc2, mc3 = st.columns(3)
        sc = "c-red" if bot_score>=70 else ("c-yellow" if bot_score>=40 else "c-green")
        ac = "c-green" if auth_score>=60 else ("c-yellow" if auth_score>=30 else "c-red")
        with mc1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Bot Probability</div><div class='metric-value {sc}'>{bot_score}%</div></div>", unsafe_allow_html=True)
        with mc2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Authenticity Score</div><div class='metric-value {ac}'>{auth_score}%</div></div>", unsafe_allow_html=True)
        with mc3:
            pc = round(profile_completeness, 0)
            pc_c = "c-green" if pc>=4 else ("c-yellow" if pc>=2 else "c-red")
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Profile Completeness</div><div class='metric-value {pc_c}'>{int(pc)}/5</div></div>", unsafe_allow_html=True)

        # Output 3 — Behavioural Anomaly Explanation
        st.markdown("<p class='sec'>⚡ Behavioural Anomaly Explanation (model-driven)</p>", unsafe_allow_html=True)
        if anomalies:
            html = "<div class='anomaly-box'>"
            for group, desc, tag_cls in anomalies:
                html += f"<span class='tag {tag_cls}'>{group}</span> {desc}<br/>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        
        if human_signals:
            st.markdown("<p class='sec'>✅ Organic Signals (pulling toward human)</p>", unsafe_allow_html=True)
            h_html = "<div class='anomaly-box' style='border-color:#4ECCA3;'>"
            for h in human_signals:
                h_html += f"✅ {h}<br/>"
            h_html += "</div>"
            st.markdown(h_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")
    st.markdown("<p style='color:#8B8FA8;'>Random Forest trained on ~37,400 real Twitter accounts. 33 behavioural features across all 4 PS indicators.</p>", unsafe_allow_html=True)
    try: m = load_metrics()
    except: st.error("Run the Colab notebook first."); st.stop()

    cols = st.columns(5)
    for col,(lbl,key) in zip(cols,[("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 Score","f1_score"),("ROC AUC","roc_auc")]):
        with col: st.markdown(f"<div class='metric-card'><div class='metric-label'>{lbl}</div><div class='metric-value c-green' style='font-size:28px;'>{m[key]}</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='metric-card' style='margin-top:12px;'><div class='metric-label'>5-Fold Cross-Validation AUC</div><div class='metric-value c-green' style='font-size:24px;'>{m.get('cv_auc_mean','?')} ± {m.get('cv_auc_std','?')}</div></div>", unsafe_allow_html=True)

    st.markdown("<p class='sec'>Visualizations</p>", unsafe_allow_html=True)
    tabs = st.tabs(["Confusion Matrix","ROC Curve","Feature Importance","Group Importance","Distributions"])
    imgs = ["outputs/confusion_matrix.png","outputs/roc_curve.png","outputs/feature_importance.png","outputs/group_importance.png","outputs/feature_distributions.png"]
    for tab, path in zip(tabs, imgs):
        with tab:
            if os.path.exists(path): st.image(path, use_container_width=True)
            else: st.info("Run the Colab notebook to generate this plot.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Dataset Explorer":
    st.markdown("# 🔬 Dataset Explorer")
    st.markdown("<p style='color:#8B8FA8;'>Real Twitter data from <b>airt-ml/twitter-human-bots</b> · HuggingFace · CC-BY-SA 3.0</p>", unsafe_allow_html=True)
    try: df = load_dataset()
    except: st.error("Run the Colab notebook first."); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,cls) in zip([c1,c2,c3,c4],[
        ("Total Records",   len(df),                "c-yellow"),
        ("Human Accounts",  (df['label']==0).sum(), "c-green"),
        ("Bot Accounts",    (df['label']==1).sum(), "c-red"),
        ("Features",        len(df.columns)-1,      "c-purple"),
    ]):
        with col: st.markdown(f"<div class='metric-card'><div class='metric-label'>{lbl}</div><div class='metric-value {cls}' style='font-size:28px;'>{val:,}</div></div>", unsafe_allow_html=True)

    DARK_BG='#0D0F14'; CARD_BG='#1A1D26'; BORDER='#2A2D3A'; GREEN='#4ECCA3'; RED='#FF4757'; GREY='#8B8FA8'

    # Feature group selector
    st.markdown("<p class='sec'>Feature Distribution Comparison</p>", unsafe_allow_html=True)
    group = st.selectbox("Feature Group", ["⏱️ Timing & Activity","💥 Burst","🌐 Network","🗣️ Linguistic"])
    group_map = {
        "⏱️ Timing & Activity": ['tweets_per_day','timing_regularity_score','lifetime_tweet_rate','tweet_density'],
        "💥 Burst":             ['burst_index','sudden_fame','engagement_paradox','abnormal_tweet_rate'],
        "🌐 Network":           ['follower_friend_ratio','friends_to_followers','network_anomaly_flag','likes_to_tweets'],
        "🗣️ Linguistic":        ['screen_name_digit_ratio','profile_completeness','description_length','bio_is_generic'],
    }
    feat = st.selectbox("Feature", group_map[group])

    fig, ax = plt.subplots(figsize=(10,4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.hist(df[df['label']==0][feat], bins=50, alpha=0.7, color=GREEN, density=True, label='Human')
    ax.hist(df[df['label']==1][feat], bins=50, alpha=0.7, color=RED,   density=True, label='Bot')
    ax.set_xlabel(feat, color=GREY, fontsize=12); ax.set_ylabel('Density', color=GREY, fontsize=12)
    ax.tick_params(colors=GREY)
    ax.legend(fontsize=11, framealpha=0.2, facecolor=CARD_BG, labelcolor='white')
    ax.grid(alpha=0.15, color=BORDER)
    st.pyplot(fig)

    st.markdown("<p class='sec'>Raw Data Preview</p>", unsafe_allow_html=True)
    filt = st.selectbox("Filter", ["All","Humans Only","Bots Only"])
    dshow = df[df['label']==0] if filt=="Humans Only" else (df[df['label']==1] if filt=="Bots Only" else df)
    st.dataframe(dshow.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — Behavioural Insights
# ══════════════════════════════════════════════════════════════════
elif page == "💡 Behavioural Insights":
    st.markdown("# 💡 Behavioural Insights")
    st.markdown("<p style='color:#8B8FA8;'>All 4 behavioural indicator groups from Problem Statement 3, with insights from the real dataset.</p>", unsafe_allow_html=True)

    groups = [
        ("⏱️ Timing Regularity", "c-red", [
            ("timing_regularity_score", "Bots post at superhuman rates relative to account age. A score >10 is a near-certain bot signal."),
            ("tweets_per_day",          "Organic users average 1–10 tweets/day. Anything above 50 is highly suspicious; above 100 is almost always a bot."),
            ("lifetime_tweet_rate",     "Measures what fraction of the account's waking hours were spent tweeting. Bots approach 24/7 saturation."),
        ]),
        ("💥 Engagement Burst Patterns", "c-yellow", [
            ("burst_index",         "A high burst index means the account posted a large volume in a short time — characteristic of coordinated bot campaigns."),
            ("sudden_fame",         "A brand-new account with thousands of followers has bought or manufactured its audience — a classic bot farm signal."),
            ("engagement_paradox",  "An account with many followers that has never liked a single post reveals its followers are also fake — nobody real is engaging."),
        ]),
        ("🌐 Network Interaction Patterns", "c-green", [
            ("follower_friend_ratio",  "Bots follow thousands but attract few genuine followers back. Extreme low ratios (<0.1) strongly indicate bot behaviour."),
            ("network_anomaly_flag",   "Flagged when an account follows >2000 people but has almost no followers — a coordinated follow-spam pattern."),
            ("likes_to_tweets",        "Real users like posts — this reciprocity ratio is near-zero for bots that only broadcast, never engage."),
        ]),
        ("🗣️ Linguistic Consistency", "c-purple", [
            ("screen_name_digit_ratio", "Bot names like 'user_48291847' have high digit ratios. Human names use words. A ratio >0.4 is a strong signal."),
            ("profile_completeness",   "Bots skip profile setup — no bio, no location, default picture, default theme. Score <2 is highly suspicious."),
            ("bio_is_generic",         "A bio shorter than 15 characters is often a bot placeholder. Real users write meaningful descriptions."),
        ]),
    ]

    for group_name, color_cls, features_info in groups:
        color = {"c-red":"#FF4757","c-yellow":"#FFE66D","c-green":"#4ECCA3","c-purple":"#A78BFA"}[color_cls]
        st.markdown(f"<div style='border-left:3px solid {color};padding-left:16px;margin:20px 0 10px 0;'><span style='font-family:monospace;font-weight:700;color:{color};font-size:17px;'>{group_name}</span></div>", unsafe_allow_html=True)
        cols = st.columns(len(features_info), gap="medium")
        for col, (fname, explanation) in zip(cols, features_info):
            with col:
                st.markdown(f"""<div class='metric-card' style='text-align:left;min-height:140px;'>
                    <div style='font-family:monospace;font-size:11px;color:{color};margin-bottom:8px;'>{fname.replace('_',' ').upper()}</div>
                    <div style='color:#C0C4D4;font-size:13px;line-height:1.6;'>{explanation}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<p class='sec'>Feature Importance by Group</p>", unsafe_allow_html=True)
    if os.path.exists("outputs/group_importance.png"):
        st.image("outputs/group_importance.png", use_container_width=True)
    st.markdown("<p class='sec'>Top 20 Feature Importance</p>", unsafe_allow_html=True)
    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png", use_container_width=True)
