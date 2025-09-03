// Legacy CLI runner kept for reference; server.js is the primary entry now.
import axios from "axios";
import fs from "fs";
import pdf from "pdf-parse/lib/pdf-parse.js";
import OpenAI from "openai";
import readlineSync from "readline-sync";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

dotenv.config();

let openai;
let pinecone;
let embeddingExtractor; // local embeddings
let qaPipeline; // local question-answering

const useCloud = Boolean(
  process.env.OPENAI_API_KEY && process.env.PINECONE_API_KEY && process.env.PINECONE_INDEX
);

// Local in-memory vector store for no-key mode
const localVectors = [];

const pdfUrl = process.env.PDF_URL || "https://raw.githubusercontent.com/yourusername/yourrepo/main/dbms.pdf";
const filePath = "document.pdf";

function assertEnv() {
  if (!useCloud) return; // local mode—no required cloud envs
  const required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"];
  const missing = required.filter((k) => !process.env[k]);
  if (missing.length) {
    throw new Error(`Missing required env vars: ${missing.join(", ")}`);
  }
}

async function downloadPDF(url, path) {
  const response = await axios.get(url, { responseType: "arraybuffer" });
  fs.writeFileSync(path, response.data);
  console.log("✅ PDF downloaded:", path);
}

async function extractText(path) {
  const dataBuffer = fs.readFileSync(path);
  const data = await pdf(dataBuffer);
  return data.text;
}

function chunkText(text, size = 800, overlap = 160) {
  const chunks = [];
  if (!text) return chunks;
  for (let i = 0; i < text.length; i += size - overlap) {
    const chunk = text.slice(i, i + size).trim();
    if (chunk) chunks.push(chunk);
  }
  return chunks;
}

async function createEmbedding(text) {
  if (useCloud) {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
    });
    return response.data[0].embedding;
  }
  // Local embedding using MiniLM (384-dim)
  if (!embeddingExtractor) {
    embeddingExtractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  const output = await embeddingExtractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function storeEmbeddings(chunks) {
  if (useCloud) {
    const indexName = process.env.PINECONE_INDEX;
    const index = pinecone.Index(indexName);
    const batch = [];
    for (let i = 0; i < chunks.length; i++) {
      const text = chunks[i];
      const embedding = await createEmbedding(text);
      batch.push({ id: `chunk-${i}`, values: embedding, metadata: { text } });
      if (batch.length >= 50) {
        await index.upsert(batch);
        batch.length = 0;
      }
    }
    if (batch.length) {
      await index.upsert(batch);
    }
    console.log("✅ Stored", chunks.length, "chunks in Pinecone.");
    return;
  }
  // Local storage
  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i];
    const embedding = await createEmbedding(text);
    localVectors.push({ id: `chunk-${i}`, values: embedding, metadata: { text } });
  }
  console.log("✅ Stored", localVectors.length, "chunks locally (no API keys).");
}

async function queryRAG(question) {
  const embedding = await createEmbedding(question);
  if (useCloud) {
    const indexName = process.env.PINECONE_INDEX;
    const index = pinecone.Index(indexName);
    const query = await index.query({ topK: 3, vector: embedding, includeMetadata: true });
    const context = (query.matches || [])
      .map((m) => (m.metadata && m.metadata.text) || "")
      .filter(Boolean)
      .join("\n\n");
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a helpful assistant that answers using the given context when relevant. If the context is insufficient, say you don't know." },
        { role: "user", content: `Question: ${question}\n\nContext:\n${context}` },
      ],
      temperature: 0.2,
    });
    return completion.choices[0]?.message?.content?.trim();
  }
  // Local: cosine similarity over normalized vectors (dot product)
  const scores = localVectors.map((v) => ({
    id: v.id,
    text: v.metadata.text,
    score: dot(embedding, v.values),
  }));
  scores.sort((a, b) => b.score - a.score);
  const top = scores.slice(0, 3);
  const context = top.map((t) => String(t.text || "")).join("\n\n");
  if (!qaPipeline) {
    qaPipeline = await pipeline("question-answering", "Xenova/distilbert-base-uncased-distilled-squad");
  }
  try {
    const result = await qaPipeline({ question: String(question || ""), context: String(context || "") });
    return result?.answer || result?.text || "(no answer)";
  } catch (e) {
    console.warn("QA pipeline failed, returning context snippet.", e?.message || e);
    return context.slice(0, 600) || "(no answer)";
  }
}

function dot(a, b) {
  let s = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) s += a[i] * b[i];
  return s;
}

(async () => {
  try {
    assertEnv();
    if (useCloud) {
      openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
      console.log("🌐 Using cloud mode (OpenAI + Pinecone)");
    } else {
      console.log("🖥️  Using local mode (no API keys): MiniLM embeddings + QA model");
    }

    await downloadPDF(pdfUrl, filePath);
    const text = await extractText(filePath);
    const chunks = chunkText(text);
    await storeEmbeddings(chunks);

    while (true) {
      const question = readlineSync.question("\n❓ Ask something (or type 'exit'): ");
      if (question.toLowerCase() === "exit") break;
      const answer = await queryRAG(question);
      console.log("\n🤖 Answer:", answer || "(no answer)");
    }
  } catch (err) {
    console.error("❌ Error:", err.message || err);
    process.exit(1);
  }
})();


