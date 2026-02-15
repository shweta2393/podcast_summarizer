import React, { useState, useRef } from "react";
import { Upload, Link, FileAudio } from "lucide-react";

function UploadSection({ onFileUpload, onUrlSubmit, disabled }) {
  const [activeTab, setActiveTab] = useState("upload");
  const [url, setUrl] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUploadClick = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleUrlSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlSubmit(url.trim());
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <section className="upload-section">
      <div className="tab-bar">
        <button
          className={`tab ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
          disabled={disabled}
        >
          <Upload size={18} />
          Upload File
        </button>
        <button
          className={`tab ${activeTab === "url" ? "active" : ""}`}
          onClick={() => setActiveTab("url")}
          disabled={disabled}
        >
          <Link size={18} />
          Paste URL
        </button>
      </div>

      <div className="tab-content">
        {activeTab === "upload" ? (
          <div className="upload-tab">
            <div
              className={`drop-zone ${dragActive ? "drag-active" : ""} ${disabled ? "disabled" : ""}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => !disabled && fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept=".mp3,.wav,.m4a,.ogg,.flac,.webm"
                hidden
                disabled={disabled}
              />
              <FileAudio size={48} className="drop-icon" />
              <p className="drop-text">
                Drag & drop your podcast audio here
              </p>
              <p className="drop-subtext">or click to browse files</p>
              <p className="drop-formats">
                Supported formats: MP3, WAV, M4A, OGG, FLAC, WEBM
              </p>
            </div>

            {selectedFile && (
              <div className="selected-file">
                <div className="file-info">
                  <FileAudio size={20} />
                  <span className="file-name">{selectedFile.name}</span>
                  <span className="file-size">
                    {formatFileSize(selectedFile.size)}
                  </span>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleUploadClick}
                  disabled={disabled}
                >
                  Summarize
                </button>
              </div>
            )}
          </div>
        ) : (
          <form className="url-tab" onSubmit={handleUrlSubmit}>
            <div className="url-input-group">
              <Link size={20} className="url-icon" />
              <input
                type="url"
                className="url-input"
                placeholder="Paste a YouTube or Spotify podcast URL..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={disabled}
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={disabled || !url.trim()}
            >
              Summarize
            </button>
            <p className="url-hint">
              Supports YouTube videos, YouTube Music, and Spotify podcast episodes
            </p>
          </form>
        )}
      </div>
    </section>
  );
}

export default UploadSection;
