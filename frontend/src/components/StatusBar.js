import React from "react";

function StatusBar({ message }) {
  return (
    <div className="status-bar">
      <div className="spinner" />
      <p className="status-message">{message}</p>
    </div>
  );
}

export default StatusBar;
