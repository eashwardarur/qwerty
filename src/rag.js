import fs from "fs";
import path from "path";
// Import internal to avoid debug behavior
import pdf from "pdf-parse/lib/pdf-parse.js";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { pipeline } from "@xenova/transformers";

// Global state
let useCloud = false;
let openai;
let pinecone;
let embeddingExtractor; // local embeddings
let qaPipeline; // local QA

// In-memory vectors for local mode
const localVectors = [];

export function configureModeFromEnv(env) {
  useCloud = Boolean(env.OPENAI_API_KEY && env.PINECONE_API_KEY && env.PINECONE_INDEX);
  return useCloud;
}

export function initCloudClients(env) {
  if (!useCloud) return;
  openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
  pinecone = new Pinecone({ apiKey: env.PINECONE_API_KEY });
}

export async function extractText(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);
  return data.text || "";
}

export function chunkText(text, size = 800, overlap = 160) {
  const chunks = [];
  const content = String(text || "");
  if (!content) return chunks;
  for (let i = 0; i < content.length; i += size - overlap) {
    const chunk = content.slice(i, i + size).trim();
    if (chunk) chunks.push(chunk);
  }
  return chunks;
}

export async function createEmbedding(text) {
  if (useCloud) {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
    });
    return response.data[0].embedding;
  }
  if (!embeddingExtractor) {
    embeddingExtractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  const output = await embeddingExtractor(String(text || ""), { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

export async function upsertChunks(chunks, namespace = undefined) {
  if (useCloud) {
    const indexName = process.env.PINECONE_INDEX;
    const index = pinecone.Index(indexName);
    const batch = [];
    for (let i = 0; i < chunks.length; i++) {
      const text = chunks[i];
      const embedding = await createEmbedding(text);
      batch.push({ id: `${namespace || "chunk"}-${i}`, values: embedding, metadata: { text } });
      if (batch.length >= 50) {
        await index.upsert(batch);
        batch.length = 0;
      }
    }
    if (batch.length) await index.upsert(batch);
    return { stored: chunks.length };
  }
  // local
  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i];
    const embedding = await createEmbedding(text);
    localVectors.push({ id: `${namespace || "chunk"}-${i}`, values: embedding, metadata: { text } });
  }
  return { stored: chunks.length };
}

export async function ingestPDF(filePath) {
  const abs = path.resolve(filePath);
  const text = await extractText(abs);
  const chunks = chunkText(text);
  return upsertChunks(chunks, path.basename(filePath));
}

export async function ingestManyPaths(paths) {
  let total = 0;
  for (const p of paths) {
    const { stored } = await ingestPDF(p);
    total += stored;
  }
  return { totalStored: total };
}

export async function ask(question, topK = 3) {
  const embedding = await createEmbedding(question);
  if (useCloud) {
    const index = pinecone.Index(process.env.PINECONE_INDEX);
    const query = await index.query({ topK, vector: embedding, includeMetadata: true });
    const context = (query.matches || [])
      .map((m) => (m.metadata && m.metadata.text) || "")
      .filter(Boolean)
      .join("\n\n");
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Answer using the given context when relevant. If insufficient, say you don't know." },
        { role: "user", content: `Question: ${question}\n\nContext:\n${context}` },
      ],
      temperature: 0.2,
    });
    return { answer: completion.choices[0]?.message?.content?.trim() || "", context };
  }
  // local mode
  const scores = localVectors.map((v) => ({ id: v.id, text: v.metadata.text, score: dot(embedding, v.values) }));
  scores.sort((a, b) => b.score - a.score);
  const top = scores.slice(0, topK);
  const context = top.map((t) => String(t.text || "")).join("\n\n");
  if (!qaPipeline) {
    qaPipeline = await pipeline("question-answering", "Xenova/distilbert-base-uncased-distilled-squad");
  }
  try {
    const result = await qaPipeline({ question: String(question || ""), context: String(context || "") });
    const answer = result?.answer || result?.text || "";
    return { answer, context };
  } catch (e) {
    return { answer: context.slice(0, 600), context };
  }
}

export function listLocalPdfCandidates(searchDirs = [process.cwd()]) {
  const files = [];
  for (const dir of searchDirs) {
    for (const name of fs.readdirSync(dir)) {
      if (name.toLowerCase().endsWith(".pdf")) files.push(path.join(dir, name));
    }
  }
  return files;
}

function dot(a, b) {
  let s = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) s += a[i] * b[i];
  return s;
}


