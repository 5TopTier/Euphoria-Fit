import os
import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN:
    raise ValueError("Missing DISCORD_TOKEN in .env")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

citation_store = {}
MEMORY_FILE = "memory.json"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


user_memory = load_memory()

SYSTEM_PROMPT = """
IDENTITY
You are the Euphoria Fit Research Bot. Your purpose is to answer
health and fitness questions by synthesizing the best available
peer-reviewed evidence — not the opinion of any single organization,
trainer, or methodology.

You have been provided with abstracts from PubMed retrieved in real
time for the user's question. Base your response on these abstracts.
Do not make claims that go beyond what the retrieved research supports.

You may also receive lightweight user context such as recurring goals,
experience level, and recent topic history. Use that only to make the
answer more relevant and less repetitive. Do not invent personal facts.

RESPONSE RULES
- Prioritize meta-analyses and systematic reviews above all else
- Prefer RCTs over observational studies
- If evidence is weak, say so directly
- Never present a single study as definitive
- If studies conflict, explain why they may conflict
- Do not diagnose injuries, conditions, or symptoms
- Do not recommend medications or therapeutic doses
- For medical questions, recommend consulting a physician or physical therapist
- Plain language, intellectually honest, no hype

STYLE RULES
You will be given a RESPONSE_STYLE:
- concise: short, direct, practical
- deep_dive: more detail, more nuance
- myth_check: directly evaluate a claim
- coaching: practical application emphasis
- conflict_analysis: compare conflicting findings clearly

FORMAT
1. WHAT THE RESEARCH SHOWS
2. EVIDENCE QUALITY
3. CONFLICTING FINDINGS (only if needed)
4. PRACTICAL TAKEAWAY
5. Want the full citations? Type !cite
""".strip()


def get_user_profile(user_id: str):
    if user_id not in user_memory:
        user_memory[user_id] = {
            "goals": [],
            "experience_level": "unknown",
            "recent_topics": [],
            "recent_styles": [],
            "last_questions": [],
            "updated_at": None,
        }
    return user_memory[user_id]


def update_user_memory(user_id: str, question: str, topic: str, style: str):
    profile = get_user_profile(user_id)

    q = question.lower()

    if any(word in q for word in ["beginner", "new to lifting", "just started", "newbie"]):
        profile["experience_level"] = "beginner"
    elif any(word in q for word in ["advanced", "peaking", "periodization", "elite", "intermediate"]):
        profile["experience_level"] = "advanced"

    goal_map = {
        "fat_loss": ["fat loss", "lose weight", "cut", "diet", "calorie deficit"],
        "muscle_gain": ["hypertrophy", "build muscle", "muscle gain", "bulk"],
        "strength": ["strength", "powerlifting", "1rm", "squat", "bench", "deadlift"],
        "recovery": ["recovery", "soreness", "sleep", "fatigue", "deload"],
        "health": ["health", "blood sugar", "blood pressure", "cholesterol", "longevity"],
    }

    for goal, keywords in goal_map.items():
        if any(k in q for k in keywords) and goal not in profile["goals"]:
            profile["goals"].append(goal)

    if topic:
        profile["recent_topics"].append(topic)
        profile["recent_topics"] = profile["recent_topics"][-8:]

    if style:
        profile["recent_styles"].append(style)
        profile["recent_styles"] = profile["recent_styles"][-5:]

    profile["last_questions"].append(question[:200])
    profile["last_questions"] = profile["last_questions"][-5:]
    profile["updated_at"] = datetime.utcnow().isoformat()

    save_memory(user_memory)


def classify_topic(question: str) -> str:
    q = question.lower()

    rules = {
        "hypertrophy": [
            "hypertrophy", "muscle growth", "build muscle", "volume",
            "sets per week", "failure training"
        ],
        "strength": [
            "strength", "1rm", "powerlifting", "squat", "bench", "deadlift"
        ],
        "fat_loss": [
            "fat loss", "lose fat", "lose weight", "cut", "deficit", "appetite"
        ],
        "nutrition": [
            "protein", "carbs", "fats", "meal timing", "calories", "diet"
        ],
        "supplements": [
            "creatine", "caffeine", "beta alanine", "supplement", "pre workout"
        ],
        "recovery": [
            "recovery", "sleep", "soreness", "fatigue", "deload", "rest day"
        ],
        "cardio": [
            "cardio", "running", "vo2", "zone 2", "aerobic", "hiit"
        ],
        "injury_pain": [
            "pain", "injury", "hurt", "strain", "sprain", "tendon", "physical therapy"
        ],
        "body_composition": [
            "body fat", "lean mass", "recomp", "body composition"
        ],
    }

    for topic, keywords in rules.items():
        if any(k in q for k in keywords):
            return topic

    return "general_fitness"


def choose_response_style(question: str, profile: dict) -> str:
    q = question.lower()

    if any(x in q for x in ["myth", "is it true", "does x really", "debunk", "fake", "bro science"]):
        return "myth_check"

    if any(x in q for x in ["conflict", "mixed evidence", "studies disagree", "vs", "better than"]):
        return "conflict_analysis"

    if any(x in q for x in ["how should i", "what should i do", "practically", "best way to apply"]):
        return "coaching"

    if any(x in q for x in ["explain deeply", "deep dive", "detailed", "thorough"]):
        return "deep_dive"

    recent = profile.get("recent_styles", [])
    if recent and recent[-1] == "concise":
        return "coaching"

    return "concise"


def build_pubmed_query(question: str, topic: str) -> str:
    topic_boosts = {
        "hypertrophy": '(muscle hypertrophy OR resistance training OR training volume)',
        "strength": '(maximal strength OR resistance training OR one repetition maximum)',
        "fat_loss": '(fat loss OR body weight OR calorie deficit OR appetite)',
        "nutrition": '(dietary protein OR energy intake OR meal timing OR sports nutrition)',
        "supplements": '(creatine OR caffeine OR ergogenic aids OR dietary supplements)',
        "recovery": '(sleep OR fatigue OR recovery OR muscle soreness)',
        "cardio": '(aerobic exercise OR HIIT OR endurance OR VO2max)',
        "injury_pain": '(musculoskeletal pain OR injury OR rehabilitation OR physical therapy)',
        "body_composition": '(body composition OR lean mass OR fat mass)',
        "general_fitness": '(exercise OR training OR fitness)',
    }

    boost = topic_boosts.get(topic, '(exercise OR training OR fitness)')
    return f"({question}) AND {boost}"


def search_pubmed(query, max_results=8):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    enhanced_query = f"{query} AND (systematic[sb] OR meta-analysis[pt] OR randomized controlled trial[pt])"

    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": enhanced_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "datetype": "pdat",
        "reldate": 3650,
    }

    try:
        search_resp = requests.get(search_url, params=search_params, timeout=20)
        search_resp.raise_for_status()
        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if len(pmids) < 3:
            search_params["term"] = query
            search_resp = requests.get(search_url, params=search_params, timeout=20)
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return [], []

        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }

        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=20)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)

        abstracts = []
        citations = []

        for article in root.findall(".//PubmedArticle"):
            abstract_texts = article.findall(".//AbstractText")
            abstract_parts = []
            for t in abstract_texts:
                text = "".join(t.itertext()).strip()
                if text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown Title"

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text.strip() if journal_el is not None and journal_el.text else "Unknown Journal"

            year = "Unknown Year"
            year_el = article.find(".//PubDate/Year")
            medline_date_el = article.find(".//PubDate/MedlineDate")
            if year_el is not None and year_el.text:
                year = year_el.text.strip()
            elif medline_date_el is not None and medline_date_el.text:
                year = medline_date_el.text.strip()

            authors = article.findall(".//Author")
            author_names = []
            for author in authors[:3]:
                last = author.find("LastName")
                if last is not None and last.text:
                    author_names.append(last.text.strip())

            author_str = ", ".join(author_names) if author_names else "Unknown authors"
            if len(authors) > 3 and author_names:
                author_str += " et al."

            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
            pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            if abstract:
                abstracts.append(
                    f"STUDY: {title}\n"
                    f"JOURNAL: {journal} ({year})\n"
                    f"FINDINGS: {abstract[:1400]}"
                )
                citations.append(f'{author_str}. "{title}" {journal}. {year}. {pubmed_link}')

        return abstracts, citations

    except Exception:
        return [], []


def build_context_block(profile: dict, topic: str, style: str) -> str:
    goals = ", ".join(profile.get("goals", [])) if profile.get("goals") else "none recorded"
    recent_topics = ", ".join(profile.get("recent_topics", [])[-3:]) if profile.get("recent_topics") else "none"
    experience = profile.get("experience_level", "unknown")

    return f"""
USER CONTEXT
- Experience level: {experience}
- Recurring goals: {goals}
- Recent topics: {recent_topics}
- Current classified topic: {topic}
- RESPONSE_STYLE: {style}

Use this only to make the answer more relevant and less repetitive.
Do not mention this context explicitly unless directly useful.
""".strip()


def synthesize_with_ai(question, abstracts, profile, topic, style):
    if not abstracts:
        research_context = "No relevant PubMed abstracts were found for this question."
    else:
        research_context = "\n\n".join(abstracts)

    context_block = build_context_block(profile, topic, style)

    user_message = f"""
{context_block}

USER QUESTION:
{question}

RETRIEVED PUBMED ABSTRACTS:
{research_context}

Additional instructions:
- Match the requested RESPONSE_STYLE.
- Avoid repeating wording used in the user's recent answers.
- If evidence is limited, say so clearly.
- If this is an injury or symptom question, do not diagnose.

Answer now.
""".strip()

    response = client.responses.create(
        model="gpt-5.2",
        instructions=SYSTEM_PROMPT,
        input=user_message,
    )

    return (response.output_text or "").strip()


async def send_long_message(channel, text, limit=1900):
    if not text:
        await channel.send("I could not generate a response.")
        return

    chunks = [text[i:i + limit] for i in range(0, len(text), limit)]
    for chunk in chunks:
        await channel.send(chunk)


async def handle_question(message_channel, user_id: str, question: str, store_key: str):
    profile = get_user_profile(user_id)
    topic = classify_topic(question)
    style = choose_response_style(question, profile)
    pubmed_query = build_pubmed_query(question, topic)

    abstracts, citations = search_pubmed(pubmed_query)
    response_text = synthesize_with_ai(question, abstracts, profile, topic, style)

    citation_store[store_key] = citations
    update_user_memory(user_id, question, topic, style)

    await send_long_message(message_channel, response_text)


@bot.event
async def on_ready():
    print(f"Euphoria Fit Research Bot is online as {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    if message.channel.name != "ask-the-science":
        await bot.process_commands(message)
        return

    if len(message.content.strip()) < 8:
        await bot.process_commands(message)
        return

    async with message.channel.typing():
        store_key = f"{message.channel.id}_{message.author.id}"
        await handle_question(
            message.channel,
            str(message.author.id),
            message.content.strip(),
            store_key,
        )

    await bot.process_commands(message)


@bot.command(name="ask")
async def ask_command(ctx, *, question: str):
    async with ctx.channel.typing():
        store_key = f"{ctx.channel.id}_{ctx.author.id}"
        await handle_question(
            ctx.channel,
            str(ctx.author.id),
            question.strip(),
            store_key,
        )


@bot.command(name="cite")
async def cite(ctx):
    store_key = f"{ctx.channel.id}_{ctx.author.id}"
    citations = citation_store.get(store_key, [])

    if not citations:
        await ctx.send("No recent citations found. Ask a question first, then type !cite.")
        return

    citation_text = "**Sources from your last question:**\n\n"
    for i, citation in enumerate(citations, 1):
        citation_text += f"{i}. {citation}\n\n"

    await send_long_message(ctx.channel, citation_text)


@bot.command(name="profile")
async def profile_command(ctx):
    profile = get_user_profile(str(ctx.author.id))
    goals = ", ".join(profile.get("goals", [])) if profile.get("goals") else "none"
    topics = ", ".join(profile.get("recent_topics", [])[-5:]) if profile.get("recent_topics") else "none"

    msg = (
        f"**Your current bot profile**\n"
        f"- Experience level: {profile.get('experience_level', 'unknown')}\n"
        f"- Goals: {goals}\n"
        f"- Recent topics: {topics}"
    )
    await ctx.send(msg)


@bot.command(name="resetprofile")
async def reset_profile(ctx):
    user_id = str(ctx.author.id)
    user_memory[user_id] = {
        "goals": [],
        "experience_level": "unknown",
        "recent_topics": [],
        "recent_styles": [],
        "last_questions": [],
        "updated_at": None,
    }
    save_memory(user_memory)
    await ctx.send("Your stored bot profile was reset.")


bot.run(DISCORD_TOKEN)
