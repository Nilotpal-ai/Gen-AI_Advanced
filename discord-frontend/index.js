require("dotenv").config();

const { Client, GatewayIntentBits } = require("discord.js");

// Node 18+ has fetch built-in (you are on v24)
const BOT_TOKEN = process.env.DISCORD_BOT_TOKEN;
const RAG_API_URL = process.env.RAG_API_URL;

if (!BOT_TOKEN || !RAG_API_URL) {
  console.error("❌ Missing environment variables");
  process.exit(1);
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

/**
 * Fired once when bot is ready
 */
client.once("ready", () => {
  console.log(`✅ Logged in as ${client.user.tag}`);
});

/**
 * Listen for messages
 */
client.on("messageCreate", async (message) => {
  try {
    // Ignore bots
    if (message.author.bot) return;

    const userQuery = message.content.trim();
    if (!userQuery) return;

    console.log(`📩 Query from ${message.author.username}: ${userQuery}`);

    await message.channel.sendTyping();

    const response = await fetch(RAG_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: userQuery }),
    });

    if (!response.ok) {
      throw new Error(`RAG API error: ${response.status}`);
    }

    const data = await response.json();
    const answer = data.answer;

    // ✅ Case 1: Not found in document
    if (!answer || answer.trim() === "") {
      await message.reply(
        "📄 This question is not covered in the Motor Insurance Handbook."
      );
      return;
    }

    // ✅ Case 2: Valid answer
    await message.reply(answer);

  } catch (err) {
    console.error("❌ Error handling message:", err);

    // Backend failure
    if (err.message.includes("500")) {
      await message.reply(
        "⚠️ The system had trouble processing this query. Please try asking a specific question from the handbook."
      );
    } else {
      await message.reply(
        "🚨 Sorry, something went wrong while processing your question."
      );
    }
  }
});

/**
 * Login
 */
client.login(BOT_TOKEN);
