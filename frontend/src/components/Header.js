import React from "react";
import { Headphones } from "lucide-react";

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <Headphones size={32} />
          <h1>Podcast Summarizer</h1>
        </div>
        <p className="tagline">
          Upload a podcast or paste a link — get an AI-powered summary in minutes
        </p>
      </div>
    </header>
  );
}

export default Header;
