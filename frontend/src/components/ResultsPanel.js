import React, { useState } from "react";
import {
  FileText,
  List,
  Sparkles,
  ScrollText,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Copy,
  Check,
  Music,
} from "lucide-react";

function ResultsPanel({ results, onReset }) {
  const [activeResultTab, setActiveResultTab] = useState("summary");
  const [showTranscript, setShowTranscript] = useState(false);
  const [copied, setCopied] = useState(false);

  const { summary, key_points, highlights, transcript, content_type } = results;
  const isSong = content_type === "song";

  const handleCopy = () => {
    const heading = isSong ? "SONG ANALYSIS" : "PODCAST SUMMARY";
    const kpLabel = isSong ? "Themes & Patterns" : "Key Points";
    const hlLabel = isSong ? "Memorable Lines" : "Highlights";

    let text = `${heading}\n${"=".repeat(50)}\n\n`;
    text += `Summary:\n${summary}\n\n`;
    text += `${kpLabel}:\n${key_points.map((p, i) => `${i + 1}. ${p}`).join("\n")}\n\n`;
    text += `${hlLabel}:\n${highlights.map((h, i) => `${i + 1}. ${h}`).join("\n")}`;

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const tabs = [
    { id: "summary", label: "Summary", icon: FileText },
    {
      id: "keypoints",
      label: isSong ? "Themes & Patterns" : "Key Points",
      icon: isSong ? Music : List,
    },
    {
      id: "highlights",
      label: isSong ? "Memorable Lines" : "Highlights",
      icon: Sparkles,
    },
  ];

  return (
    <section className="results-panel">
      <div className="results-header">
        <div className="results-title-group">
          <h2>Results</h2>
          {isSong && <span className="content-type-badge">Song Detected</span>}
        </div>
        <div className="results-actions">
          <button
            className="btn btn-icon"
            onClick={handleCopy}
            title="Copy all results"
          >
            {copied ? <Check size={18} /> : <Copy size={18} />}
            {copied ? "Copied!" : "Copy"}
          </button>
          <button className="btn btn-secondary" onClick={onReset}>
            <RotateCcw size={18} />
            New Summary
          </button>
        </div>
      </div>

      <div className="results-tabs">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`results-tab ${activeResultTab === id ? "active" : ""}`}
            onClick={() => setActiveResultTab(id)}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      <div className="results-content">
        {activeResultTab === "summary" && (
          <div className="result-card">
            <p className="summary-text">{summary}</p>
          </div>
        )}

        {activeResultTab === "keypoints" && (
          <div className="result-card">
            {key_points.length > 0 ? (
              <ol className="key-points-list">
                {key_points.map((point, index) => (
                  <li key={index} className="key-point-item">
                    <span className="point-number">{index + 1}</span>
                    <span className="point-text">{point}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="no-data">
                {isSong
                  ? "No themes or patterns detected."
                  : "No key points extracted."}
              </p>
            )}
          </div>
        )}

        {activeResultTab === "highlights" && (
          <div className="result-card">
            {highlights.length > 0 ? (
              <div className="highlights-list">
                {highlights.map((highlight, index) => (
                  <blockquote key={index} className="highlight-item">
                    <p>"{highlight}"</p>
                  </blockquote>
                ))}
              </div>
            ) : (
              <p className="no-data">
                {isSong
                  ? "No memorable lines extracted."
                  : "No highlights extracted."}
              </p>
            )}
          </div>
        )}
      </div>

      {transcript && (
        <div className="transcript-section">
          <button
            className="transcript-toggle"
            onClick={() => setShowTranscript(!showTranscript)}
          >
            <ScrollText size={18} />
            <span>{isSong ? "Full Lyrics" : "Full Transcript"}</span>
            {showTranscript ? (
              <ChevronUp size={18} />
            ) : (
              <ChevronDown size={18} />
            )}
          </button>
          {showTranscript && (
            <div className="transcript-content">
              <p>{transcript}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default ResultsPanel;
