"""Curated fixed dataset of Hindi and Hinglish customer-support utterances.

Includes failure-mode tags:
- clean: Baseline customer support queries
- code_mixing: Heavy Hindi-English mixing with domain-specific terms (OTP, KYC, refund, etc.)
- regional_accent: Dialectal phrasing and colloquial regional inflections (Bhojpuri, Hyderabadi, Punjabi, etc.)
- background_noise: Realistic on-the-go queries with simulated acoustic noise
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    lang: str  # "hi" or "hi-en"
    domain: str  # ecommerce, banking, telecom, food_delivery, travel
    failure_modes: List[str] = field(default_factory=list)


UTTERANCES: List[Utterance] = [
    # -------------------------------------------------------------
    # 1. Clean Baseline (10 utterances)
    # -------------------------------------------------------------
    Utterance(
        id="utt_001",
        text="mera refund kab tak aayega",
        lang="hi",
        domain="banking",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_002",
        text="order cancel karne ka process bataiye",
        lang="hi",
        domain="ecommerce",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_003",
        text="customer care officer se baat karni hai",
        lang="hi",
        domain="telecom",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_004",
        text="delivery partner ka number chahiye",
        lang="hi",
        domain="food_delivery",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_005",
        text="mera account statement email par bhej do",
        lang="hi",
        domain="banking",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_006",
        text="flight ticket ka status confirm hai ya nahi",
        lang="hi",
        domain="travel",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_007",
        text="registered mobile number badalna chahta hoon",
        lang="hi",
        domain="telecom",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_008",
        text="kya return pickup aaj ho sakta hai",
        lang="hi",
        domain="ecommerce",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_009",
        text="address update karne mein madad chahiye",
        lang="hi",
        domain="food_delivery",
        failure_modes=["clean"],
    ),
    Utterance(
        id="utt_010",
        text="mera bill payment receipt nahi mila",
        lang="hi",
        domain="telecom",
        failure_modes=["clean"],
    ),

    # -------------------------------------------------------------
    # 2. Code-Mixing (10 utterances)
    # -------------------------------------------------------------
    Utterance(
        id="utt_011",
        text="mera payment deduct ho gaya but booking confirm nahi hui",
        lang="hi-en",
        domain="travel",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_012",
        text="app login karte waqt OTP receive nahi ho raha hai",
        lang="hi-en",
        domain="banking",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_013",
        text="order dispatch ho chuka hai ya tracking link update hoga",
        lang="hi-en",
        domain="ecommerce",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_014",
        text="credit card limit increase karne ke liye KYC update kaise karein",
        lang="hi-en",
        domain="banking",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_015",
        text="food deliver nahi hua aur app par delivered show kar raha hai",
        lang="hi-en",
        domain="food_delivery",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_016",
        text="broadband speed bahut slow hai connection reset kar do please",
        lang="hi-en",
        domain="telecom",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_017",
        text="cash on delivery order ko online payment mein convert karna hai",
        lang="hi-en",
        domain="ecommerce",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_018",
        text="flight reschedule charges kitne lagenge customer support agent",
        lang="hi-en",
        domain="travel",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_019",
        text="wallet balance freeze ho gaya hai unblock kaise hoga",
        lang="hi-en",
        domain="banking",
        failure_modes=["code_mixing"],
    ),
    Utterance(
        id="utt_020",
        text="subscription auto renew ho gaya please refund process initiate kijiye",
        lang="hi-en",
        domain="telecom",
        failure_modes=["code_mixing"],
    ),

    # -------------------------------------------------------------
    # 3. Regional Accents / Dialectal Inflections (10 utterances)
    # -------------------------------------------------------------
    Utterance(
        id="utt_021",
        text="humra account se paisa kat gail ba abhi tak refund naikhe aail",
        lang="hi",
        domain="banking",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_022",
        text="ka ho bhaiya hamar parcel kab le pahunchi delivery wala",
        lang="hi",
        domain="food_delivery",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_023",
        text="mera parcel kidhar atka hua hai yaaro subah se wait kar rahe",
        lang="hi",
        domain="ecommerce",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_024",
        text="agent se connect karo na bhai kitna der hold pe rakhte",
        lang="hi",
        domain="telecom",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_025",
        text="paise debit ho gaye par ticket koni mili re bhai",
        lang="hi",
        domain="travel",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_026",
        text="order cancel kar do ji refund kado tak aayega hamare account vich",
        lang="hi",
        domain="ecommerce",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_027",
        text="saar bill zyada aaya hai connection bandh karna mangta hai",
        lang="hi",
        domain="telecom",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_028",
        text="ekdum thik se batao mera khana kab tak aayega delivery boy rasta bhool gaya",
        lang="hi",
        domain="food_delivery",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_029",
        text="sirji bank balance check nahi ho raha server down bata raha hai kya",
        lang="hi",
        domain="banking",
        failure_modes=["regional_accent"],
    ),
    Utterance(
        id="utt_030",
        text="humko abhi ke abhi manager se baat karao bahut preshani ho raha hai",
        lang="hi",
        domain="telecom",
        failure_modes=["regional_accent"],
    ),

    # -------------------------------------------------------------
    # 4. Background Noise (10 utterances)
    # -------------------------------------------------------------
    Utterance(
        id="utt_031",
        text="hello aawaz aa rahi hai kya mera cab booking cancel karna hai",
        lang="hi",
        domain="travel",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_032",
        text="bhaiya raste mein hoon delivery address note kar lijiye landmark mandir hai",
        lang="hi",
        domain="food_delivery",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_033",
        text="atm machine mein card phans gaya hai jaldi card block kijiye",
        lang="hi",
        domain="banking",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_034",
        text="metro station pe khada hoon ticket download nahi ho rahi",
        lang="hi",
        domain="travel",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_035",
        text="market mein bahut shor hai mera order cancel karke refund initiate karo",
        lang="hi",
        domain="ecommerce",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_036",
        text="signal drop ho raha hai broadband connection ka complaint number de do",
        lang="hi",
        domain="telecom",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_037",
        text="hello hello delivery agent sun nahi raha call disconnect mat karna",
        lang="hi",
        domain="food_delivery",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_038",
        text="upi pin bhool gaya hoon account reset karne ka OTP bhejiye",
        lang="hi",
        domain="banking",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_039",
        text="package damaged mila hai return pickup request raise kar dijiye",
        lang="hi",
        domain="ecommerce",
        failure_modes=["background_noise"],
    ),
    Utterance(
        id="utt_040",
        text="network problem hai internet bilkul chal nahi raha technician bhejo",
        lang="hi",
        domain="telecom",
        failure_modes=["background_noise"],
    ),
]


def get_utterance_by_id(utt_id: str) -> Optional[Utterance]:
    """Retrieve an utterance by its unique ID."""
    for u in UTTERANCES:
        if u.id == utt_id:
            return u
    return None


def filter_utterances_by_tag(tag: str) -> List[Utterance]:
    """Filter utterances that have a specific failure mode tag."""
    return [u for u in UTTERANCES if tag in u.failure_modes]
