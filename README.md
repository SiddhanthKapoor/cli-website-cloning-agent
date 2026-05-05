# Assignment 02 — AI Agent CLI Tool

This is my submission for Assignment 02 from the GenAI Engineering course. The task was to build a CLI agent that can take natural language instructions and actually do things — kind of like how Cursor or Windsurf work, but much simpler obviously.

We covered AI agents in Session 3, and the whole idea of *why* LLMs need tools (since they can't fetch live data on their own, can't write files, etc.) clicked for me while building this. The agent uses the **ReAct pattern** we studied — Reasoning + Acting — where it alternates between thinking about what to do and actually doing it.

---

## What it does

You type something in the terminal, the agent figures out how to handle it step by step, and gives you a result. It's not limited to one thing — you can ask it:

- What's the weather in some city
- Info about a GitHub user
- Create a simple web app or page
- Clone the Scaler Academy website (the main demo for this assignment)
- List files, run commands, read/write files

The main demo task from the assignment is cloning the Scaler website with a Header, Hero section, and Footer — and the agent does this by generating an HTML file and opening it in the browser automatically.

---

## How it works — the loop

This is basically the ReAct loop from class. The agent doesn't do everything in one shot — it loops through steps:

```
You type something
        ↓
   THINK — agent figures out what to do next
        ↓
   TOOL — it calls a tool (write file, run command, fetch weather, etc.)
        ↓
   OBSERVE — it sees what the tool returned
        ↓
   (repeat until done)
        ↓
   OUTPUT — final answer / result
```

Each step, the model returns a single JSON object like this:

```json
{ "step": "THINK", "content": "I need to create the folder first" }
{ "step": "TOOL", "tool_name": "executeCommand", "tool_args": { "cmd": "mkdir -p ./scaler_clone" } }
{ "step": "OBSERVE", "content": "Command executed successfully." }
{ "step": "OUTPUT", "content": "Done! The file is open in your browser." }
```

The system prompt tells the model exactly what format to follow, what tools it has, and the rules (one step at a time, wait for OBSERVE before the next tool, etc.).

---

## Tools the agent has access to

| Tool | What it does |
|------|-------------|
| `getWeather` | Gets live weather for any city using wttr.in |
| `getGitHubUser` | Fetches public info about a GitHub user |
| `executeCommand` | Runs a shell command on your machine |
| `writeFile` | Writes content to a file (creates the folder too if needed) |
| `appendFile` | Appends content to an existing file |
| `readFile` | Reads a file |
| `listFiles` | Lists what's in a directory |
| `openInBrowser` | Opens a local file in the browser |
| `fetchWebpage` | **(NEW)** Converts any live URL into clean Markdown using the Jina Reader API so the agent can read real websites without crashing the context window |

The weather and GitHub tools are things I added myself — they were in the reference code in the assignment PDF and made sense to keep in since it shows the agent isn't just a "website builder", it can actually handle different kinds of tasks.

---

## Setting it up

You need Node.js. This agent uses a **Multi-Provider Waterfall system**. You can provide keys for Groq, OpenRouter, and Gemini all at once. The agent will automatically use the fastest free model (Groq) and fallback instantly if it hits rate limits.

```bash
git clone <your-repo-url>
cd scaler-cli-agent
npm install
```

Create a `.env` file and add ANY or ALL of these:
```
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=AIza...
```

Then just run it:
```bash
node agent.js
```

---

## Running it — what you'll see

```
╔══════════════════════════════════════════════════════════╗
║   🚀  Scaler CLI Agent  ·  Groq + OpenRouter + Gemini  ║
║   ReAct Loop : THINK → TOOL → OBSERVE → OUTPUT           ║
╚══════════════════════════════════════════════════════════╝

You can ask me anything! Examples:
  • "What's the weather in Mumbai?"
  • "Show me GitHub info for torvalds"
  • "Create a todo app with HTML, CSS and JS"
  • "Clone the Scaler Academy website with Header, Hero and Footer"

You: Clone the Scaler Academy website with Header, Hero section and Footer

🤖 Agent started...

💭 THINK: The user wants to clone the Scaler website. I should read the live site first to get the content.
🔧 TOOL: fetchWebpage
   📋 OBSERVE: # Become the Professional Built for the Next Decade in AI...
💭 THINK: I'll now generate the clone based on this content.
🔧 TOOL: writeFile
   📋 OBSERVE: File written: /path/to/scaler_clone/index.html (18500 bytes)
🔧 TOOL: listFiles
   📋 OBSERVE: 📄 index.html
🔧 TOOL: openInBrowser
   📋 OBSERVE: Opened in browser: /path/to/scaler_clone/index.html

✅ OUTPUT:
The Scaler Academy clone is ready and open in your browser!
```

---

## The Scaler clone itself

The generated page has:

- **Header** — logo, nav links (Courses, Mentors, Events, Blog), a Login button and a "Get Started" CTA
- **Hero section** — main headline, subtext, two CTAs, and a stats bar (number of students, hiring partners, salary hike)
- **Courses section** — 4 cards for DSA, System Design, Full Stack, and Data Science — each with a badge, duration, and enroll button
- **Footer** — links, social icons, copyright

Colors are based on the actual Scaler brand — dark background (`#0D0D1A`), their blue/indigo (`#3D3AEE`), and orange accent (`#FF6B35`). Fonts loaded via Google Fonts inside the `<style>` block so it's fully self-contained.

---

## Files in this project

```
scaler-cli-agent/
├── agent.js          ← the main file — ReAct loop + Multi-Provider + all tools
├── package.json
├── .env              ← your API key goes here (not committed)
├── .gitignore
├── README.md
└── scaler_clone/
    └── index.html    ← generated when you run the clone task
```

---

## Packages used

- `openai` — to call Groq and OpenRouter models seamlessly
- `@google/generative-ai` — to call the Gemini fallback models
- `axios` — for HTTP requests (weather, GitHub, Jina Reader API)
- `dotenv` — to load the `.env` file
- `open` — to open the generated HTML in the browser

---

## What I learned building this

1. **Context Limits and Rate Limits are brutal:** Originally, I chunked file writing, which caused the agent to loop 15+ times. The loop hit Groq's 30 requests-per-minute limit instantly. Then, the prompt size grew too large and hit a `413 Request too large` error. The fix was forcing the model to dump the entire website in a *single* `writeFile` call, and keeping the agent's short-term memory capped to the last 6 actions.
2. **Reading websites is hard for LLMs:** Telling an LLM to `curl scaler.com` dumps 200KB of raw HTML tags into the prompt, crashing it instantly. I integrated the `Jina Reader API` (via `fetchWebpage` tool) which elegantly turns any live URL into clean Markdown so the LLM can actually read the website safely before cloning it.
3. **Multi-provider fallbacks:** Free APIs go down all the time. Building a flat queue of models (Groq -> OpenRouter -> Gemini) that auto-skips on 429/503 errors makes the agent incredibly robust.
