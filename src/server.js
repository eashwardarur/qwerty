import express from "express";
import cors from "cors";
import path from "path";
import dotenv from "dotenv";
import {
  configureModeFromEnv,
  initCloudClients,
  ingestManyPaths,
  listLocalPdfCandidates,
  ask,
} from "./rag.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const PORT = process.env.PORT || 3000;

const useCloud = configureModeFromEnv(process.env);
if (useCloud) initCloudClients(process.env);

// Static frontend
app.use(express.static(path.join(process.cwd(), "public")));

app.post("/api/ingest", async (req, res) => {
  try {
    const { paths } = req.body || {};
    const targetPaths = Array.isArray(paths) && paths.length ? paths : listLocalPdfCandidates([process.cwd()]);
    const { totalStored } = await ingestManyPaths(targetPaths);
    res.json({ ok: true, totalStored, paths: targetPaths });
  } catch (e) {
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

app.post("/api/ask", async (req, res) => {
  try {
    const { question, topK } = req.body || {};
    if (!question) return res.status(400).json({ ok: false, error: "Missing 'question'" });
    const result = await ask(question, topK || 3);
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

app.listen(PORT, async () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  try {
    const toIngest = listLocalPdfCandidates([process.cwd()]);
    if (toIngest.length) {
      const { totalStored } = await ingestManyPaths(toIngest);
      console.log(`📚 Auto-ingested ${totalStored} chunks from ${toIngest.length} PDFs.`);
    } else {
      console.log("ℹ️  No local PDFs found to auto-ingest.");
    }
  } catch (e) {
    console.warn("Auto-ingest failed:", e?.message || e);
  }
});


