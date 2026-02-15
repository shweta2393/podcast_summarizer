import React, { useState } from "react";
import Header from "./components/Header";
import UploadSection from "./components/UploadSection";
import StatusBar from "./components/StatusBar";
import ResultsPanel from "./components/ResultsPanel";
import { uploadAudio, submitUrl, pollJobStatus } from "./api";
import "./App.css";

function App() {
  const [status, setStatus] = useState("idle"); // idle, uploading, processing, completed, failed
  const [statusMessage, setStatusMessage] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const resetState = () => {
    setStatus("idle");
    setStatusMessage("");
    setResults(null);
    setError("");
  };

  const handlePoll = async (jobId) => {
    const statusMap = {
      downloading: "Downloading audio from URL...",
      processing: "Preparing audio for transcription...",
      transcribing: "Transcribing audio with Whisper AI...",
      summarizing: "Generating summary and key points...",
    };

    let attempts = 0;
    const maxAttempts = 600; // 30 min max

    while (attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const job = await pollJobStatus(jobId);
        setStatusMessage(statusMap[job.status] || `Status: ${job.status}`);

        if (job.status === "completed") {
          setResults(job.result);
          setStatus("completed");
          return;
        }
        if (job.status === "failed") {
          setError(job.error || "Processing failed.");
          setStatus("failed");
          return;
        }
      } catch (e) {
        setError("Lost connection to server.");
        setStatus("failed");
        return;
      }
      attempts++;
    }
    setError("Processing timed out.");
    setStatus("failed");
  };

  const handleFileUpload = async (file) => {
    resetState();
    setStatus("uploading");
    setStatusMessage("Uploading audio file...");
    try {
      const { job_id } = await uploadAudio(file);
      setStatus("processing");
      setStatusMessage("Processing started...");
      await handlePoll(job_id);
    } catch (e) {
      setError(e.response?.data?.detail || "Upload failed. Please try again.");
      setStatus("failed");
    }
  };

  const handleUrlSubmit = async (url) => {
    resetState();
    setStatus("uploading");
    setStatusMessage("Submitting URL...");
    try {
      const { job_id } = await submitUrl(url);
      setStatus("processing");
      setStatusMessage("Downloading audio...");
      await handlePoll(job_id);
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to process URL.");
      setStatus("failed");
    }
  };

  const isProcessing = status === "uploading" || status === "processing";

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <UploadSection
          onFileUpload={handleFileUpload}
          onUrlSubmit={handleUrlSubmit}
          disabled={isProcessing}
        />

        {isProcessing && <StatusBar message={statusMessage} />}

        {status === "failed" && (
          <div className="error-banner">
            <p>{error}</p>
            <button onClick={resetState} className="btn btn-secondary">
              Try Again
            </button>
          </div>
        )}

        {status === "completed" && results && (
          <ResultsPanel results={results} onReset={resetState} />
        )}
      </main>

      <footer className="footer">
        <p>Powered by Whisper AI &amp; BART &mdash; Built with React &amp; FastAPI</p>
      </footer>
    </div>
  );
}

export default App;
